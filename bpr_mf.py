import torch
import torch.nn as nn
import os
import random
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
from time import time
from util_wjy.parser import parse_args
from util_wjy.read_data import *
from util_wjy.evaluate import evaluate
from scipy.sparse import csr_matrix


class BPRMF(nn.Module):

    def __init__(self, userNum, itemNum):
        super(BPRMF, self).__init__()
        self.n_users = userNum
        self.n_items = itemNum
        self.device = args.device
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.batch_size = args.batch_size

        self.decay = eval(args.regs)[0]

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

    def forward(self, users, pos_items, neg_items, _, drop_flag=False):
        user_embedding = self.embedding_dict['user_emb'][users, :]
        pos_i_embedding = self.embedding_dict['item_emb'][pos_items, :]
        neg_i_embedding = self.embedding_dict['item_emb'][neg_items, :]
        return user_embedding, pos_i_embedding, neg_i_embedding, _

    def rating(self, user_embedding, pos_i_embedding):
        return torch.matmul(user_embedding, pos_i_embedding.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)


        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss


if __name__ == "__main__":
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(123456)
    torch.manual_seed(123456)
    
    # ======= read data =======
    dataset_name = "filmtrust"
    train_percent = "70"
    tuning_num = '1'
    folder = "/home/wjysys/ssrPytorchtest/dataset"
    train_file = f"{dataset_name}_train_{train_percent}_{tuning_num}.csv"
    test_file = f"{dataset_name}_test_{train_percent}_{tuning_num}.csv"

    train_path = os.path.join(folder, dataset_name, train_file)
    test_path = os.path.join(folder, dataset_name, test_file)

    print(f"train_path: {train_path}")
 
    args.lr = 0.1
    args.embed_size = 128
    args.batch_size = 1024
    args.regs = '[0.1]'

    data_generator = Data(args.batch_size, train_path, test_path)
    
    model = BPRMF(data_generator.n_users,
                  data_generator.n_items).to(args.device)
    t0 = time()
    #Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_loger, pre_loger, rec_loger, ndcg_loger, map_loger = [], [], [], [], []

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_bpr_mf()
            users = torch.LongTensor(users).cuda()
            pos_items = torch.LongTensor(pos_items).cuda()
            neg_items = torch.LongTensor(neg_items).cuda()
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, _ = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           [])

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
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
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, pres_temp[0], pres_temp[-1],
                        recs_temp[0], recs_temp[-1], maps_temp[0], maps_temp[-1], ndcgs_temp[0], ndcgs_temp[-1])
            print(perf_str)

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