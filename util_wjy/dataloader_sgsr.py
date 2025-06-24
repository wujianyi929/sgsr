# import cupy as cp
# from cupyx.scipy.sparse import csr_matrix
import numpy as np
import torch
from scipy.sparse import csr_matrix
import random as rd
from util_wjy.parser import parse_args


args = parse_args()


class Data(object):
    def __init__(self, batch_size, train_path, test_path, trust_path=''):
        self.batch_size = batch_size
        self.device = args.device
        self.beta = args.beta
        self.k = args.k
        
        _X = np.loadtxt(train_path, delimiter=",", dtype=int)
        _Y = np.loadtxt(test_path, delimiter=",", dtype=int)
        
        self.n_users = max(np.max(_X[:, 0]), np.max(_Y[:, 0]))
        self.n_items = max(np.max(_X[:, 1]), np.max(_Y[:, 1]))
        self.n_train = _X.shape[0]
        self.n_test = _Y.shape[0]
        self.g_size = args.group_size
        
        self.train_set, self.test_set = {}, {}
        self.trustee_set = {}
        self.trustGraph = None
        if trust_path:  # 若有trust数据集
            _T = np.loadtxt(trust_path, delimiter=" ", dtype=int)
            sn_max = max(np.max(_T[:, 0]), np.max(_T[:, 1]))
            self.n_users = max(self.n_users, sn_max)
            _T -= 1
            trusters = _T[:, 0]
            trustees = _T[:, 1]
            csr_T = csr_matrix((np.ones(_T.shape[0]), (_T[:, 0], _T[:, 1])), shape=(self.n_users, self.n_users))
            # self.trustWeightGraph = self.getWeightGraph_small(trusters, trustees, self.n_users, self.n_users, True)
            if not args.l1:
                print("Use sqrt normalization.")
                self.trustWeightGraph = self.getWeightGraph_trust(trusters, trustees, self.n_users, True)
            else:
                print("Use l1 normalization.")
                self.trustWeightGraph = self.getWeightGraph_trust_l1(trusters, trustees, self.n_users, True)
            
            print('Loaded trustWeightGraph!')
                   
        _X -= 1
        _Y -= 1
        self.all_items = set(range(self.n_items))
        self.neg_set = {}
        self.train_item2user = {}
        train_items = _X[:, 1]
        test_items = _Y[:, 1]
        train_users = _X[:, 0]
        test_users = _Y[:, 0]
        trustors = _T[:, 0]
        trustees = _T[:, 1]
        
        # 最后一个参数True，表示选取自身的交互数量设置一个权重
        if not args.l1:
            self.uiWeightGraph = self.getWeightGraph_small(train_users, train_items, self.n_users, self.n_items, False, False)
        else:
            self.uiWeightGraph = self.getWeightGraph_small_l1(train_users, train_items, self.n_users, self.n_items, False, False)
        print('Loaded uiWeightGraph!')
        
        csr_X = csr_matrix((np.ones(_X.shape[0]), (train_users, train_items)),
                           shape=(self.n_users, self.n_items))
        csr_Y = csr_matrix((np.ones(_Y.shape[0]), (test_users, test_items)),
                           shape=(self.n_users, self.n_items))
        
        self.get_ui_dict(train_users, csr_X, self.train_set, True, self.neg_set)
        self.get_ui_dict(train_items, csr_X.T, self.train_item2user)
        self.get_ui_dict(test_users, csr_Y, self.test_set)
        self.get_ui_dict(trustors, csr_T, self.trustee_set)
        
        self.exist_users = list(self.train_set.keys())
        self.n_exist_users = len(self.exist_users)
        self.n_test_users = len(self.test_set.keys())

        self.print_statistics()
    
    def getWeightGraph_trust_l1(self, trusters, trustees, n_user, getSelfNorm):
        truster_dim = torch.LongTensor(trusters)
        trustee_dim = torch.LongTensor(trustees)
        
        index = torch.stack([truster_dim, trustee_dim])
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, torch.Size([n_user, n_user]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = D.unsqueeze(dim=0)
        dense = dense/D_sqrt
        if getSelfNorm:
            dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([n_user, n_user]))
        
        Graph = Graph.coalesce().to(args.device)
        return Graph
    
    def getWeightGraph_small_l1(self, node1s, node2s, n_node1, n_node2, get2Sub, getSelfNorm):
        node1_dim = torch.LongTensor(node1s)
        node2_dim = torch.LongTensor(node2s)
        # 如果按照这个方法计算的话，实际上是算的用户自身的出度，以及其信任对象的入度
        first_sub = torch.stack([node1_dim, node2_dim + n_node1])
        second_sub = torch.stack([node2_dim + n_node1, node1_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, torch.Size([n_node1+n_node2, n_node1+n_node2]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = D.unsqueeze(dim=0)
        dense = dense/D_sqrt
        if getSelfNorm:
            dense = dense/D_sqrt.t()
        
        if get2Sub:
            dense_s = dense[:n_node1, n_node1:]
            sparse_shape0 = n_node1
            sparse_shape1 = n_node2
        else:
            dense_s = dense[n_node1:, :n_node1]
            sparse_shape0 = n_node2
            sparse_shape1 = n_node1
        index = dense_s.nonzero()
        data  = dense_s[dense_s >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([sparse_shape0, sparse_shape1]))
        
        Graph = Graph.coalesce().to(args.device)
        return Graph
    
    def getWeightGraph_trust(self, trusters, trustees, n_user, getSelfNorm):
        truster_dim = torch.LongTensor(trusters)
        trustee_dim = torch.LongTensor(trustees)
        
        index = torch.stack([truster_dim, trustee_dim])
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, torch.Size([n_user, n_user]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        if getSelfNorm:
            dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([n_user, n_user]))
        
        Graph = Graph.coalesce().to(args.device)
        return Graph
    
    def getWeightGraph_small(self, node1s, node2s, n_node1, n_node2, get2Sub, getSelfNorm):
        node1_dim = torch.LongTensor(node1s)
        node2_dim = torch.LongTensor(node2s)
        # 如果按照这个方法计算的话，实际上是算的用户自身的出度，以及其信任对象的入度
        first_sub = torch.stack([node1_dim, node2_dim + n_node1])
        second_sub = torch.stack([node2_dim + n_node1, node1_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, torch.Size([n_node1+n_node2, n_node1+n_node2]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        if getSelfNorm:
            dense = dense/D_sqrt.t()
        
        if get2Sub:
            dense_s = dense[:n_node1, n_node1:]
            sparse_shape0 = n_node1
            sparse_shape1 = n_node2
        else:
            dense_s = dense[n_node1:, :n_node1]
            sparse_shape0 = n_node2
            sparse_shape1 = n_node1
        index = dense_s.nonzero()
        data  = dense_s[dense_s >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([sparse_shape0, sparse_shape1]))
        
        Graph = Graph.coalesce().to(args.device)
        return Graph
    
    def getWeightGraph(self, node1s, node2s, n_node1, n_node2, flag1):
        user_dim = torch.LongTensor(node1s)
        node2_dim = torch.LongTensor(node2s)
        
        first_sub = torch.stack([user_dim, node2_dim + n_node1])
        second_sub = torch.stack([node2_dim + n_node1, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()
        Graph = torch.sparse.IntTensor(index, data, torch.Size([n_node1+n_node2, n_node1+n_node2]))
        dense = Graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D==0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense/D_sqrt
        if flag1:
            dense = dense/D_sqrt.t()
        index = dense.nonzero()
        data  = dense[dense >= 1e-9]
        assert len(index) == len(data)
        Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([n_node1+n_node2, n_node1+n_node2]))
        Graph = Graph.coalesce().to(args.device)
        
        return Graph
    
    def get_ui_dict(self, keys, csr_mat: csr_matrix, dict: dict, flag=False, neg_dic: dict=None):
        for id in np.unique(keys):
            vals = csr_mat[id].nonzero()[1]
            torch_vals = torch.tensor(vals, device=self.device)  # 转到gpu上再转为list，采样效率会高一些
            dict[id] = torch_vals.tolist()
            if flag:
                pos_set = set(dict[id])
                neg_dic[id] = list(self.all_items - pos_set)
        
    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample_pos_items_for_u(self, u):
        # sample 1 pos items for u-th user
        pos_items = self.train_set[u]
        n_pos_items = len(pos_items)
        pos_id = rd.randint(0, n_pos_items-1)
        pos_i_id = pos_items[pos_id]
        return pos_i_id
    
    def sample_sgsr(self):
        if self.batch_size <= self.n_exist_users:  # self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
           
        def sample_group_users(u, item, num):
            # sample num group members for u-th user's i-th interaction
            group_users = {u}
            len_i2u = len(self.train_item2user[item])
            if len_i2u <= num:
                return self.train_item2user[item]
            while True:
                if len(group_users) == num:
                    break
                guser_id = rd.sample(self.train_item2user[item], num-len(group_users))
                group_users.update(guser_id)
            group_users = list(group_users)

            return group_users
                    
        def sample_group_mask(u, item, num):
            # sample num group members for u-th user's i-th interaction
            group_users = {u}
            len_i2u = len(self.train_item2user[item])
            if len_i2u <= num:
                group_users = self.train_item2user[item]
            else:
                while True:
                    if len(group_users) == num:
                        break
                    guser_id = rd.sample(self.train_item2user[item], num-len(group_users))
                    group_users.update(guser_id)
                group_users = list(group_users)
            
            mask = torch.zeros(self.n_users, device=self.device)
            mask[group_users] = 1.

            return mask
            
        def sample_neg_items_N(u, num):
            # sample num neg items for u-th user
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = rd.randint(0, self.n_items-1)
                if neg_id not in self.train_set[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_N1(u, num):
            # sample num neg items for u-th user
            neg_items = rd.sample(self.neg_set[u], num)
            return neg_items
        
        pos_items, group_users, neg_items_N = [], [], []
        group_masks = []
        for u in users:
            item = self.sample_pos_items_for_u(u)
            pos_items.append(item)
            group_masks.append(sample_group_mask(u, item, args.group_size))
            neg_items_N.append(sample_neg_items_N1(u, 5))
        
        group_mat = torch.stack(group_masks)
        
        return users, pos_items, neg_items_N, group_mat
    
    def sample_sgsr_new1(self):
        if self.batch_size <= self.n_exist_users:  # self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        
        def sample_group_mask(u, item, num):
            # sample num group members for u-th user's i-th interaction
            group_users = {u}
            len_i2u = len(self.train_item2user[item])
            if len_i2u <= num:
                group_users = self.train_item2user[item]
            else:
                while True:
                    if len(group_users) == num:
                        break
                    guser_id = rd.sample(self.train_item2user[item], num-len(group_users))
                    group_users.update(guser_id)
                group_users = list(group_users)
                
            return group_users

        def sample_neg_items_N1(u, num):
            # sample num neg items for u-th user
            neg_items = rd.sample(self.neg_set[u], num)
            return neg_items
        
        pos_items, neg_items_N = [], []
        # group_masks = []
        g_masks = torch.zeros((self.batch_size, self.n_users), device=self.device)
        i = 0
        for u in users:
            item = self.sample_pos_items_for_u(u)
            pos_items.append(item)
            g_users = sample_group_mask(u, item, args.group_size)
            g_masks[i][g_users] = 1.
            neg_items_N.append(sample_neg_items_N1(u, 5))
            i += 1
        return users, pos_items, neg_items_N, g_masks
    
    def sample_sgsr_nog(self):
        if self.batch_size <= self.n_exist_users:  # self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        
        def sample_neg_items_N1(u, num):
            # sample num neg items for u-th user
            neg_items = rd.sample(self.neg_set[u], num)
            return neg_items
        
        pos_items, neg_items_N = [], []
        g_masks = torch.zeros((self.batch_size, self.n_users), device=self.device)
        i = 0
        for u in users:
            item = self.sample_pos_items_for_u(u)
            pos_items.append(item)
            neg_items_N.append(sample_neg_items_N1(u, 5))
            i += 1
        return users, pos_items, neg_items_N, g_masks
    
    def sample_sgsr_new2(self):
        '''优化了group mask的采样,以及采样group时不考虑是否包含自身'''
        if self.batch_size <= self.n_exist_users:  # self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        
        def sample_group_mask(i, item, num, gmask):
            # sample num group members for u-th user's i-th interaction
            if len(self.train_item2user[item]) <= num:
                group_users = self.train_item2user[item]
            else:
                group_users = rd.sample(self.train_item2user[item], num)
                
            gmask[i][group_users] = 1.

        def sample_neg_items_N1(u, num):
            # sample num neg items for u-th user
            neg_items = rd.sample(self.neg_set[u], num)
            return neg_items
        
        pos_items, neg_items_N = [], []
        # group_masks = []
        g_masks = torch.zeros((self.batch_size, self.n_users), device=self.device)
        i = 0
        for u in users:
            item = self.sample_pos_items_for_u(u)
            pos_items.append(item)
            sample_group_mask(i, item, 3, g_masks)
            neg_items_N.append(sample_neg_items_N1(u, 5))
            i += 1
        return users, pos_items, neg_items_N, g_masks
      
    def sample_sgsr_sg(self):
        if self.batch_size <= self.n_exist_users:  # self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        
        def sample_social_mask(u, num):
            s_users = {u}
            len_su = len(self.trustee_set.get(u, []))
            if len_su <= num:
                s_users = self.trustee_set.get(u, [])
            else:
                while True:
                    if len(s_users) == num:
                        break
                    s_uid = rd.sample(self.trustee_set.get(u, []), num-len(s_users))
                    s_users.update(s_uid)
                s_users = list(s_users)
            
            mask = torch.zeros(self.n_users, device=self.device)
            mask[s_users] = 1.
            
            return mask 
                    
        def sample_group_mask(u, item, num):
            # sample num group members for u-th user's i-th interaction
            group_users = {u}
            len_i2u = len(self.train_item2user[item])
            if len_i2u <= num:
                group_users = self.train_item2user[item]
            else:
                while True:
                    if len(group_users) == num:
                        break
                    guser_id = rd.sample(self.train_item2user[item], num-len(group_users))
                    group_users.update(guser_id)
                group_users = list(group_users)
            
            mask = torch.zeros(self.n_users, device=self.device)
            mask[group_users] = 1.

            return mask

        def sample_neg_items_N1(u, num):
            # sample num neg items for u-th user
            neg_items = rd.sample(self.neg_set[u], num)
            return neg_items
        
        pos_items, neg_items_N = [], []
        group_masks = []
        social_masks = []
        for u in users:
            item = self.sample_pos_items_for_u(u)
            pos_items.append(item)
            group_masks.append(sample_group_mask(u, item, 3))
            social_masks.append(sample_social_mask(u, 3))
            neg_items_N.append(sample_neg_items_N1(u, 5))
        
        group_mat = torch.stack(group_masks)
        social_mat = torch.stack(social_masks)
        return users, pos_items, neg_items_N, group_mat, social_mat
    
    
    # def sample_neg_items_N1(self, u, num):
    #         # sample num neg items for u-th user
    #         neg_items = rd.sample(self.neg_set[u], num)
    #         return neg_items
    
    # def sample_sgsr_withoutG(self):
    #     if self.batch_size <= self.n_users:
    #         users = rd.sample(self.exist_users, self.batch_size)
    #     else:
    #         users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]
        
    #     pos_items, neg_items_N = [], []
    #     for u in users:
    #         item = self.sample_pos_items_for_u(u)
    #         pos_items.append(item)
    #         neg_items_N.append(self.sample_neg_items_N1(u, 5))
        
    #     return users, pos_items, neg_items_N
        
    