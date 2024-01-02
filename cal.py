import traceback
import numpy as np

def points2points_l2_norm(m1,m2):
    # return point-wise L2 distance
    # `m1` ndarray [N,D]
    # `m2` ndarray [M,D]

    N = m1.shape[0]
    M=m2.shape[0]
    m1 = np.expand_dims(m1, axis=0)
    m1 = np.repeat(m1, repeats=M, axis=0)  # [M,N,D]

    m2 = np.expand_dims(m2, axis=1)  # [M,1,D]
    m2 = np.repeat(m2, repeats=N, axis=1)  # [M,N,D]

    dist = np.sqrt(np.sum((m1 - m2) * (m1 - m2), axis=2).T)  # [N,M]

    return dist

def get_emb_fasttext(words,model):
    # return ndarray[N,D]
    emb=[]
    # get embeddings
    emb = []
    for w in words:
        emb = model.get_word_vector(w).reshape((1, -1))
        emb.append(emb)
    emb = np.concatenate(emb, axis=0)
    return emb

def find_neighbors(cands, cores, fast_model, topk, type='nearest', only_words=True):
    # INPUT:
    # `type` string "nearest" or "farest"
    # Output:
    # return word list or tuple(word,score)

    # get embeddings
    core_emb = []
    for w in cores:
        emb = fast_model.get_word_vector(w).reshape((1, -1))
        core_emb.append(emb)
    core_emb = np.concatenate(core_emb, axis=0)

    cand_emb = []
    for w in cands:
        try:
            emb = fast_model.get_word_vector(w).reshape((1, -1))
        except:
            print("#%s# failed to get embedding. " % w)
            emb = np.zeros((1, 300))
        cand_emb.append(emb)
    cand_emb = np.concatenate(cand_emb, axis=0)

    # calculate L2 distance
    dist = points2points_l2_norm(cand_emb, core_emb)

    min_d = dist.min(axis=1)

    q = []
    for w, d in zip(cands, min_d):
        q.append((w, d))

    if type=='nearest':
        q.sort(key=lambda x: x[1])

    if type=='farest':
        q.sort(key=lambda x: -x[1])

    if only_words:
        res = [x[0] for x in q]
        return res[:topk]
    else:
        return q[:topk]

def KL(prob1, prob2):
    """
    Compute KL divergence.
    :param prob1: list of prob
    :param prob2: list of prob
    :return: scalar
    """
    prob1 = np.asarray(prob1, dtype=np.float)
    prob2 = np.asarray(prob2, dtype=np.float)
    return np.sum(np.where(prob1 != 0, prob1 * np.log(prob1 / prob2), 0))

def cal(labeled_words,unlabeled_words, acq_size, w_prob_map, K, fast_model):
    """
    Single iteration of CAL. Paper: https://arxiv.org/pdf/2109.03764.pdf
    FastText word embeddings are used for neighbor-findings.
    :param labeled_words:
    :param unlabeled_words:
    :param acq_size: acquisition size
    :param w_prob_map: dict{word: prob list}
    :param K: number of neighbours
    :param fast_model: calculate word representations
    :return: selected of words, list;
    """

    res=[]
    for xp in unlabeled_words: # line 1 in "Algorithm 1" from original paper
        try:
            neighs=find_neighbors(labeled_words, [xp], fast_model, topk=K, type='nearest', only_words=True) # line 2

            kl_sum=0
            p_xp=w_prob_map[xp] # line 4
            for xl in neighs:
                p_xl = w_prob_map[xl]  # line 3
                kl_sum+=KL(p_xl, p_xp) # line 5
            kl_avg=kl_sum/len(neighs) # line 6
            res.append((xp,kl_avg))

        except:
            print(traceback.format_exc())

    res.sort(key=lambda x: -x[1]) # descending

    return res[:acq_size]  # line 8


def debug_cal(labeled_words,unlabeled_words, acq_size, w_prob_map, K, fast_model):
    """
    Single iteration of CAL. Paper: https://arxiv.org/pdf/2109.03764.pdf
    FastText word embeddings are used for neighbor-findings.
    :param labeled_words:
    :param unlabeled_words:
    :param acq_size: acquisition size
    :param w_prob_map: dict{word: prob list}
    :param K: number of neighbours
    :param fast_model: calculate word representations
    :return: selected of words, list;
    """
    d_xp_prob={} # dict{xp:prob_list}
    d_xp_neigh_prob = {} # dict{xp: [[xp_neigh, prob_list]]}

    res=[]
    for xp in unlabeled_words: # line 1 in "Algorithm 1" from original paper
        try:
            neighs=find_neighbors(labeled_words, [xp], fast_model, topk=K, type='nearest', only_words=True) # line 2
            d_xp_neigh_prob[xp]=[]
            kl_sum=0
            p_xp=w_prob_map[xp] # line 4
            for xl in neighs:
                p_xl = w_prob_map[xl]  # line 3
                d_xp_neigh_prob[xp].append([xl]+p_xl)
                kl_sum+=KL(p_xl, p_xp) # line 5
            kl_avg=kl_sum/len(neighs) # line 6
            res.append((xp,kl_avg))

            d_xp_prob[xp]=p_xp


        except:
            print(traceback.format_exc())

    res.sort(key=lambda x: -x[1]) # descending

    return res[:acq_size],d_xp_prob, d_xp_neigh_prob # line 8


if __name__=="__main__":

    # KL test
    p1=[0.2,0.8]
    p2=[0.3,0.7]
    print(KL(p1, p2))

    # cal test
    import joblib
    import fasttext as ft
    import pandas as pd
    fast_model = ft.load_model('source/cc.en.300.bin')
    all_words = pd.read_csv('Data/adj.csv')['word'].tolist()
    labeled_words = pd.read_csv('Data/train.v4.5.round5.csv')['word'].tolist()

    pred = joblib.load('Data/pred.v4.5.round5')
    w_prob_map={}
    for i,w in enumerate(all_words):
        w_prob_map[w]=[1-pred[i].item(),pred[i].item()]
    unlabeled_words=[ w for w in all_words if w not in labeled_words]

    #res=cal(labeled_words, unlabeled_words, 40, w_prob_map, 10 , fast_model)
    res, d_xp_prob, d_xp_neigh_prob  = debug_cal(labeled_words, unlabeled_words, 40, w_prob_map, 10, fast_model)

    for x in res:
        print(x)

    """
    'creamiest': [0.9475813060998917, 0.05241869390010834]
    
    labeled_neighbors:
    ['comfortable', 0.0552639365196228, 0.9447360634803772],
     ['amazing', 0.04938077926635742, 0.9506192207336426],
     ['crispy', 0.8976927250623703, 0.1023072749376297],
     ['decorative', 0.8759047910571098, 0.12409520894289017],
     ['awesome', 0.05167144536972046, 0.9483285546302795],
     ['consistent', 0.4175049662590027, 0.5824950337409973],
     ['conventional', 0.0642327070236206, 0.9357672929763794],
    ['convenient', 0.06981676816940308, 0.9301832318305969],
    ['drippy', 0.8805531114339828, 0.11944688856601715],
    ['contributive', 0.06828141212463379, 0.9317185878753662]
    
    Its neighbors are almost predicted as positive while it's predicted as negative.
    """


    exit(0)







