import cupy as cp
import numpy as np
import torch
# from scipy.sparse import csr_matrix
from cupyx.scipy.sparse import csr_matrix
import random as rd
import os
from util_wjy.parser import parse_args


args = parse_args()


class Data(object):
    def __init__(self, batch_size, train_path, test_path, trust_path=''):
        self.batch_size = batch_size
        self.beta = args.beta
        X = cp.loadtxt(train_path, delimiter=",", dtype=int)
        Y = cp.loadtxt(test_path, delimiter=",", dtype=int)
        self.n_users = max(cp.max(X[:, 0]).item(), cp.max(Y[:, 0]).item())  # 训练集和测试集中最大的用户id
        self.n_items = max(cp.max(X[:, 1]).item(), cp.max(Y[:, 1]).item())

        self.train_items, self.test_set, self.trusts = {}, {}, {}
        self.sqrt_degree_X, self.sqrt_degree_T = [], []
        if trust_path:  # 若有trust数据集
            T = cp.loadtxt(trust_path, delimiter=" ", dtype=int)
            sn_max = max(cp.max(T[:, 0]).item(), cp.max(T[:, 1]).item())  # 社交网络中最大的用户id
            self.n_users = max(self.n_users, sn_max)
            T -= 1
            t = T.astype('float32')
            csr_T = csr_matrix((cp.ones(t.shape[0]), (t[:, 0], t[:, 1])), shape=(self.n_users, self.n_users))
            temp1 = cp.sqrt(csr_T.sum(axis=1).ravel())
            self.sqrt_degree_T = [float(deg) for deg in temp1] 
            self.get_ui_dict(T[:, 0]-1, T[:, 1]-1, self.trusts)
                   
        self.n_train = X.shape[0]
        self.n_test = Y.shape[0]

        # self.k is the number of sampling
        self.k = args.k

        self.train_item2user = {}
        train_items = X[:, 1]-1
        test_items = Y[:, 1]-1
        
        train_users = X[:, 0]-1
        test_users = Y[:, 0]-1
        
        csr_X = csr_matrix((cp.ones(X.shape[0]), (self.train_users.astype('float32'), self.train_items.astype('float32'))),
                           shape=(self.n_users, self.n_items))
        temp1 = cp.sqrt(csr_X.sum(axis=1).ravel())
        self.sqrt_degree_X = [float(deg) for deg in temp1] 
        
        self.get_ui_dict(train_users, train_items, self.train_items)
        self.get_ui_dict(train_items, train_users, self.train_item2user)
        self.get_ui_dict(test_users, test_items, self.test_set)
        
        # Frequency of each item
        self.freq = {}
        self.exist_users = list(self.train_items.keys())
        self.n_test_users = len(self.test_set.keys())

        self.print_statistics()
 
    def get_ui_dict(self, keys, values, dict):
        for id in cp.unique(keys):
            ids = int(id)
            vals = values[keys == ids]
            dict[ids] = [int(i) for i in vals]
        
    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample_pos_items_for_u(self, u):
        # sample 1 pos items for u-th user
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_id = rd.randint(0, n_pos_items-1)
        pos_i_id = pos_items[pos_id]
        return pos_i_id
        
    def sample_gsetrank(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
  
        def sample_group_users(u, item, num):
            # sample num group members for u-th user's i-th interaction
            group_users = {u}
            len_i2u = len(self.train_item2user[item])
            if len_i2u < num:
                num = len_i2u
            while True:
                if len(group_users) == num:
                    break
                guser_id = rd.sample(self.train_item2user[item], num)
                group_users.update(guser_id)

            return list(group_users)

        def sample_neg_items_N(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                # neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                neg_id = rd.randint(0, self.n_items-1)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items_N = [], []
        for u in users:
            pos_items += self.sample_pos_items_for_u(u)
            neg_items_N.append(sample_neg_items_N(u, 5))
        return users, pos_items, neg_items_N
    
    def sample_sgsr(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_trusts_relation(u):
            return self.trusts.get(u, [])
           
        def sample_group_users(u, item, num):
            # sample num group members for u-th user's i-th interaction
            group_users = {u}
            len_i2u = len(self.train_item2user[item])
            if len_i2u < num:
                num = len_i2u
            while True:
                if len(group_users) == num:
                    break
                guser_id = rd.sample(self.train_item2user[item], num-len(group_users))
                group_users.update(guser_id)

            return list(group_users)
            
        def sample_neg_items_N(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = rd.randint(0, self.n_items-1)
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, trust_users, group_users, neg_items_N = [], [], [], []
        for u in users:
            item = self.sample_pos_items_for_u(u)
            pos_items.append(item)
            trust_users.append(sample_trusts_relation(u))
            group_users.append(sample_group_users(u, item, 3))
            neg_items_N.append(sample_neg_items_N(u, 5))
        return users, pos_items, trust_users, group_users, neg_items_N
        
    