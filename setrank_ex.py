import numpy as np
import random
import torch
import torch.nn as nn
import os
import torch.optim as optim
from time import time
from tqdm import tqdm
from util_wjy.parser import parse_args
from util_wjy.read_data import *
from util_wjy.evaluate import evaluate
from util_wjy import wus_tools as wst
from scipy.sparse import csr_matrix


class SetRank(nn.Module):

    def __init__(self, userNum, itemNum):
        super(SetRank, self).__init__()
        self.n_users = userNum
        self.n_items = itemNum
        self.device = args.device
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.batch_size = args.batch_size
        self.regs = eval(args.regs)[0]

        self.embedding_dict = self.init_weight()

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users,
                                                 self.embed_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items,
                                                 self.embed_size)))
        })
        return embedding_dict

    def forward(self, users, pos_items, pot_items, neg_items, drop_flag=False):
        user_embedding = self.embedding_dict['user_emb'][users, :]
        pos_i_embedding = self.embedding_dict['item_emb'][pos_items, :]
        pot_i_embedding = self.embedding_dict['item_emb'][pot_items, :]
        neg_i_embedding = self.embedding_dict['item_emb'][neg_items, :]
        return user_embedding, pos_i_embedding, pot_i_embedding, neg_i_embedding

    def rating(self, user_embedding, pos_i_embedding):
        return torch.matmul(user_embedding, pos_i_embedding.t())

    def create_list_loss(self, users, pos_items, neg_items):
        users_embeddings = self.embedding_dict['user_emb'][users, :]
        pos_embeddings = self.embedding_dict['item_emb'][pos_items, :]
        neg_embeddings = self.embedding_dict['item_emb'][neg_items, :]

        score_ui = torch.exp(torch.sum(torch.mul(users_embeddings, pos_embeddings), axis=1))
        users_embeddings_new = torch.unsqueeze(users_embeddings, dim=1).expand(neg_embeddings.size())
        score_uj = torch.sum(torch.exp(torch.sum(torch.mul(users_embeddings_new, neg_embeddings), dim=2)), dim=1)

        list_loss = torch.mean(torch.log(score_ui / (score_ui + score_uj)) * -1)

        regularizer = (torch.norm(users_embeddings) ** 2 + torch.norm(pos_embeddings) ** 2 + \
                       torch.norm(neg_embeddings) ** 2) / 2
        emb_loss = self.regs * regularizer / self.batch_size

        return list_loss + emb_loss, list_loss, emb_loss


if __name__ == "__main__":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    random.seed(123456)
    torch.manual_seed(123456)
    
    # ======= read data =======
    dataset_name = "lastfm"
    train_percent = "70"
    tuning_num = '1'
    folder = "/home/wjysys/ssrPytorchtest/dataset"
    train_file = f"{dataset_name}_train_{train_percent}_{tuning_num}.csv"
    test_file = f"{dataset_name}_test_{train_percent}_{tuning_num}.csv"

    train_path = os.path.join(folder, dataset_name, train_file)
    test_path = os.path.join(folder, dataset_name, test_file)

    print(f"train_path: {train_path}")
 
    args.lr = 0.001
    args.embed_size = 128
    args.batch_size = 1024
    args.regs = '[0.01]'

    data_generator = Data(args.batch_size, train_path, test_path)

    model = SetRank(data_generator.n_users, data_generator.n_items).to(args.device)

    t0 = time()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, map_loger = [], [], [], [], []

    for epoch in range(args.epoch):
        t1 = time()
        loss, list_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items_N = data_generator.sample_setrank()  # 尽可能的采样过程不出现numpy数据以及cpu与gpu数据的交换，影响gpu效率
            batch_loss, batch_list_loss, batch_emb_loss = model.create_list_loss(users, pos_items, neg_items_N)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            list_loss += batch_list_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, list_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        U = model.embedding_dict['user_emb'].detach().cpu().numpy()
        V = model.embedding_dict['item_emb'].detach().cpu().numpy()
        pres_temp, recs_temp, maps_temp, ndcgs_temp = evaluate(U, V, data_generator.train_set, data_generator.test_set,
                                                                   data_generator.n_test_users, [5, 10])

        t3 = time()

        loss_loger.append(loss)
        pre_loger.append(pres_temp)
        rec_loger.append(recs_temp)
        map_loger.append(maps_temp)
        ndcg_loger.append(ndcgs_temp)

        if args.verbose > 0:
            perf_str = 'Ed %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], \n' \
                       'precision=[%.5f, %.5f], recall=[%.5f, %.5f], map=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, list_loss, emb_loss, pres_temp[0], pres_temp[-1],
                        recs_temp[0], recs_temp[-1], maps_temp[0], maps_temp[-1], ndcgs_temp[0], ndcgs_temp[-1])
            print(perf_str)

        # cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
        #                                                             stopping_step, expected_order='acc', flag_step=5)
        # # *********************************************************
        # # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        # if should_stop == True:
        #     break

    pres = np.array(pre_loger)
    recs = np.array(rec_loger)
    maps = np.array(map_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\t precision=[%s], recall=[%s], map=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in maps[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
