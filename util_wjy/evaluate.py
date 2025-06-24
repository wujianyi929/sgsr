import numpy as np
import torch
from time import time

def evaluate(u_fac, v_fac, train_set, test_set, test_size, K):
    start = time()
    len_K = len(K)
    precision = np.zeros(len_K)
    recall = np.zeros(len_K)
    MAP = np.zeros(len_K)
    NDCG = np.zeros(len_K)
    # F1 = np.zeros(len_K)
    # MRR = np.zeros(len_K)
    
    test_nz = list(test_set.keys())
    score = np.dot(u_fac, v_fac.T)
    for us in test_nz:
        test_u_indices = test_set[us]
        # len_test_u = len(test_u_indices)
        test_u = set(test_u_indices)

        pre_rec_tmp = np.zeros(len_K)
        map_tmp = np.zeros(len_K)
        DCG_tmp = np.zeros(len_K)
        iDCG_tmp = np.zeros(len_K)
        # mrr_tmp = np.zeros(len_K)
        
        pos_u = train_set.get(us, -1)
        if pos_u == -1:
            continue

        score[us, pos_u] = -999
        score_u = score[us]
        # 获取 top-K 推荐的索引
        top_K_indices = np.argpartition(score_u, -K[-1])[-K[-1]:]
        top_K_scores = score_u[top_K_indices]
        sorted_indices = top_K_indices[np.argsort(top_K_scores)[::-1]]

        cc = 0
        # reciprocal_rank_recorded = False
        for c in range(1, K[-1] + 1):
            j = sorted_indices[c - 1]
            if j in test_u:
                cc += 1
                for k in range(len(K) - 1, -1, -1):
                    if c <= K[k]:
                        pre_rec_tmp[k] += 1
                        map_tmp[k] += cc / c
                        DCG_tmp[k] += 1 / np.log2(c + 1)
                        # if not reciprocal_rank_recorded:
                        #     mrr_tmp += 1 / c
                        #     reciprocal_rank_recorded = True
                    else:
                        break
                    
        test_u_items = len(test_u)
        idcg = 0
        ki = 0
        for k in range(1, K[-1] + 1):
            if k <= test_u_items:
                idcg += 1 / np.log2(k+1)
            if k in K:
                iDCG_tmp[ki] = idcg
                ki += 1
                 
        n_test = len(test_u)
        pre = pre_rec_tmp / K
        rec = pre_rec_tmp / n_test
        precision += pre  # 对应元素相除
        recall += rec
        MAP += map_tmp / test_u_items
        NDCG += DCG_tmp / iDCG_tmp
        # if (pre+rec).all():  # 所有元素都不为0，a.all()输出为True
        #     F1 += 2*pre*rec/(pre+rec) 
        # MRR += mrr_tmp

    end = time()
    print(f"evaluate time: {end-start:.4f}s")
    return precision / test_size, recall / test_size, MAP / test_size, NDCG / test_size  #, F1 / test_size, MRR / test_size