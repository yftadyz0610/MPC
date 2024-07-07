import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import joblib

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.metrics import accuracy_score

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BertModel,  BertTokenizer

#

import warnings
warnings.simplefilter('ignore')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True   # True will slow down training.

# 设置随机数种子
setup_seed(20) # [6,12,20]

root_dir='./'
bert_model_tag = 'bert-base-uncased'    
bert_model = BertModel.from_pretrained(bert_model_tag, torchscript=True)
tokenizer = BertTokenizer.from_pretrained(bert_model_tag)

class Config:
    file_path = root_dir+'Data/train_v1_r1tor7_2.csv'
    task_name = 'mpc'
    version = '1.7.2' # model_version.round_id.try_id
    epoch_num=20
    warmup=40 # warmup steps
    lr = 2e-5
    stepLR=[10]
    max_len = 256
    train_bs = 32
    valid_bs = 64
    train_pcent = 0.8
    num_workers = 0
    device=1

    resume_training=False
    last_version=''
    last_load_ckpt_path = ''

    is_predict=False # if True do only predict otherwise do only train
    test_file_path=root_dir+'Data/SentiWordNet3_new_a_v.csv'
    test_sep='|'
    load_ckpt_path=root_dir+'CkptSaveDir/%s/version_%s/epoch=6-step=55.ckpt'%(task_name, version)
    predict_output_path=root_dir+'Data/pred.v1.r7.1.npy'

class BertData(Dataset):
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = Config.max_len

    def __len__(self):
        return len(self.review)

    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = ' '.join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=Config.max_len,
            pad_to_max_length=True,
            truncation='longest_first'
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        targets = torch.tensor(self.target[idx], dtype=torch.float)

        return {'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
                }


class BERTModel(pl.LightningModule):
    def __init__(self) -> None:
        super(BERTModel, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        self.all_targets = []
        self.train_loss_fn = nn.BCEWithLogitsLoss()
        self.valid_loss_fn = nn.BCEWithLogitsLoss()
        self.epoch=0

    def forward(self, ids, mask, token_type_ids) -> torch.Tensor:
        _, output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output = self.drop(output)
        output = self.out(output)
        return output

    def prepare_data(self) -> None:
        # Load the data, encode, shuffle and split it
        data = pd.read_csv(Config.file_path)
        """data = pd.read_csv(Config.file_path, encoding='latin-1',
                           names=['target', 'id', 'date', 'query', 'username', 'text'])"""
        data = data[['target', 'text']]
        data['target'] = data['target'].map({1: 1, 0: 0}) # customized target values
        data = data.sample(frac=1).reset_index(drop=True)

        nb_training_samples = int(Config.train_pcent * len(data))

        self.train_data = data[:nb_training_samples]
        self.valid_data = data[nb_training_samples:]

        # Make Training and Validation Datasets
        self.training_set = BertData(
            review=self.train_data['text'].values,
            target=self.train_data['target'].values
        )

        self.validation_set = BertData(
            review=self.valid_data['text'].values,
            target=self.valid_data['target'].values
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.training_set,
            batch_size=Config.train_bs,
            shuffle=True,
            num_workers=Config.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.validation_set,
            batch_size=Config.valid_bs,
            shuffle=False,
            num_workers=Config.num_workers,
        )
        return val_loader

    def training_step(self, batch, batch_idx):
        ids = batch['ids'].long()
        mask = batch['mask'].long()
        token_type_ids = batch['token_type_ids'].long()
        targets = batch['targets'].float()

        outputs = self(ids=ids, mask=mask, token_type_ids=token_type_ids)
        pred_prob=torch.sigmoid(outputs)

        train_loss = self.train_loss_fn(outputs, targets.view(-1, 1))
        return {'loss': train_loss, 'logits': outputs, 'pred_prob':pred_prob, 'true_label': targets.view(-1, 1)}

    def training_epoch_end(self, outputs) -> None:
        # calculating average batch loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        y_true = torch.cat([x['true_label'] for x in outputs]).cpu().numpy()
        y_pred = torch.cat([x['logits'] for x in outputs]).cpu().numpy()
        y_pred_prob = torch.cat([x['pred_prob'] for x in outputs]).cpu().numpy()

        acc=accuracy_score(y_true,y_pred_prob>0.5)

        log_batch={"train_epoch_loss":avg_loss, "train_accuracy":acc, "step":self.epoch}
        self.log_dict(log_batch, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.epoch+=1
        return

    def validation_step(self, batch, batch_idx):
        ids = batch['ids'].long()
        mask = batch['mask'].long()
        token_type_ids = batch['token_type_ids'].long()
        targets = batch['targets'].float()

        outputs = self(ids=ids, mask=mask, token_type_ids=token_type_ids)
        pred_prob = torch.sigmoid(outputs)

        self.all_targets.extend(targets.cpu().detach().numpy().tolist())

        valid_loss = self.valid_loss_fn(outputs, targets.view(-1, 1))
        return {'val_loss': valid_loss, 'logits': outputs, 'pred_prob':pred_prob, 'true_label': targets.view(-1, 1)}

    def validation_epoch_end(self, outputs) -> None:
        # calculating average batch loss
        #
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        y_true = torch.cat([x['true_label'] for x in outputs]).cpu().numpy()
        y_pred = torch.cat([x['logits'] for x in outputs]).cpu().numpy()
        y_pred_prob = torch.cat([x['pred_prob'] for x in outputs]).cpu().numpy()

        acc = accuracy_score(y_true, y_pred_prob > 0.5)

        log_batch = {"val_epoch_loss": avg_loss, "val_accuracy": acc, "step": self.epoch}
        self.log_dict(log_batch, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return

    def predict_step(self, batch, batch_idx,):
        ids = batch['ids'].long()
        mask = batch['mask'].long()
        token_type_ids = batch['token_type_ids'].long()
        targets = batch['targets'].float()

        outputs = self(ids=ids, mask=mask, token_type_ids=token_type_ids)
        pred_prob = torch.sigmoid(outputs)
        return pred_prob


    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt=transformers.AdamW(optimizer_parameters, lr=Config.lr)
        return [opt],\
               [MultiStepLR(opt, milestones=Config.stepLR, gamma=0.1)]

    # learning rate warm-up
    def optimizer_step(
            self,
            epoch,
            batch_idx,
            optimizer,
            optimizer_idx,
            optimizer_closure,
            on_tpu=False,
            using_native_amp=False,
            using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < Config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / Config.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * Config.lr

# setup standalone dataloader for prediction
def set_dataloader(file_path,sep=','):
    data = pd.read_csv(file_path,sep)
    data = data[['target', 'text']]
    dataset=BertData(
        review=data['text'].values,
        target=data['target'].values
    )
    data_loader=DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    return data_loader


if __name__=="__main__":
    if not Config.is_predict:
        # Run the training loop

        if Config.resume_training:
            model = BERTModel.load_from_checkpoint(checkpoint_path=Config.last_load_ckpt_path)
        else:
            model = BERTModel()

        # tensorboard logger
        logger = TensorBoardLogger(save_dir=os.path.join(root_dir, 'TbDir'), name=Config.task_name, version=Config.version)
        # tensorboard --logdir=tb_dir

        ckpt_save_path = 'CkptSaveDir/' + '%s/' % Config.task_name + 'version_%s' % Config.version
        # checkpoint setting
        checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(root_dir, ckpt_save_path), save_top_k=3,
                                              mode='min', monitor="val_epoch_loss")

        trainer = pl.Trainer(max_epochs=Config.epoch_num, accelerator='gpu', devices=[Config.device], logger=logger,
                             callbacks=[checkpoint_callback], log_every_n_steps=10,)
        trainer.fit(model)
        print("Tensorboard check command: tensorboard --logdir=%s"%os.path.join(root_dir, 'tb_dir'))

        print("Ckpt is saved at: %s"%os.path.join(root_dir,ckpt_save_path))

    else:
        dl=set_dataloader(Config.test_file_path,Config.test_sep)
        model = BERTModel.load_from_checkpoint(checkpoint_path=Config.load_ckpt_path)
        trainer = pl.Trainer(accelerator='gpu', devices=[Config.device])
        predictions = trainer.predict(model, dataloaders=dl)
        predictions = torch.cat(predictions).cpu().numpy()
        np.save(Config.predict_output_path,predictions)
        print('Predictions are saved at: %s'%(Config.predict_output_path))
        exit(0)



















