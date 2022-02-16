import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_parent import ParentNodeLayer
from layer_agg import AggregateLayer
from layer_pred import PredLayer
from PositionalEncoder import SinusoidalEncoder

import config

class DUVRec(nn.Module):

    def __init__(self, dim_node, dim_hidden, n_heads, n_assign,\
                 num_items_all, device,\
                 module_test=False, **kwargs\
                 ):

        super(DUVRec, self).__init__()

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dim_node = dim_node

        self.parentNodeLayer = ParentNodeLayer(dim_node, dim_hidden, n_heads, n_assign, device).to(device)
        self.aggregateLayer  = AggregateLayer().to(device)
        self.predLayer       = PredLayer().to(device)       
        self.positionEncoder = SinusoidalEncoder(emb_dim=dim_node, max_len=config.dataset[config.dataset_choice]['max_position_span']).to(device)
        self.dropout_layer   = nn.Dropout(p=0.2)

        self.embedding_item = nn.Embedding(num_embeddings=num_items_all+1, embedding_dim=dim_node)
        if 1:#self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            #world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')

        self.device = device
        self.module_test = module_test

    def prepareEmbFromID(self, Embedding, *ids):
        '''input:
                Embedding     : (n_item_all, dim_node)
                history_id    : (batch_size, n_node) 
                pos_id, neg_id: (batch_size, 1)
            output:
                x  : (batch_size, n_node, dim_node)
                pos: (batch_size, 1     , dim_node)
                neg: (batch_size, n_neg     , dim_node)
        '''
        return [Embedding(i) for i in ids]


    def bpr_loss(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp: (batch_size, n_node)
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.pos: (batch_size, 1)
                candidates.neg: (batch_size, n_neg)
                candidates.stp: (batch_size, 1)
        output:
            loss, loss_reg
        '''
        x, pos, neg = self.prepareEmbFromID(self.embedding_item, users.history, candidates.pos, candidates.neg)
        cand = torch.cat((pos,neg), dim=1) # (batch_size, n_cand, dim_node)
        batch_size = cand.shape[0]
        adj = users.adj
        #x   : (batch_size, n_node, dim_node)
        #adj : (batch_size, n_node, n_node)
        #pos : (batch_size, 1     , dim_node)
        #neg : (batch_size, n_neg , dim_node)

        # add positional embedding
        x_PE = self.positionEncoder(users.timestp)
        x += x_PE 
        
        cdd_PE = self.positionEncoder(candidates.stp) # (batch_size, 1, dim_node)
        n_cand = cand.shape[1]
        cand += cdd_PE.repeat(1,n_cand,1)

        # bpr loss
        x_parent, s = self.parentNodeLayer(x, adj) #(batch_size, n_assign, dim_node), (batch_size, n_node, n_assign)
        
        scores_list = []
        for i in range(n_cand):
            user4cand = self.aggregateLayer(x_parent, cand[:,i,:].unsqueeze(1)) #(batch_size, 1, dim_node)
            scores_list.append(self.predLayer(user4cand, cand[:,i,:].unsqueeze(1))) #(batch_size, 1)
        pos_scores = scores_list[0]
        neg_scores = torch.cat(scores_list[1:], dim=1) #(batch_size, n_neg)

        pos_scores = pos_scores.repeat(1, candidates.neg.shape[1]) #(batch_size, n_neg) 
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # entropy loss
        batch_size = s.size()[0]
        n_node = s.size()[1]
        loss_entropy = (-torch.sum(s*torch.log(s)))/(batch_size*n_node)

        # reg_loss
        batch_size = x.size()[0]
        loss_reg = (1/2)*(x.norm(2).pow(2) +
                        pos.norm(2).pow(2) +
                        neg.norm(2).pow(2))/float(batch_size)
        
        # con_loss
        # cand (batch_size, n_cand, dim_node)
        user_emb_I = self.dropout_layer(torch.mean(x, dim=1)) #(batch_size, dim_node)
        user_emb_F = self.dropout_layer(torch.mean(x_parent, dim=1)) #(batch_size, dim_node)
        scores_I = torch.matmul(cand, user_emb_I.unsqueeze(-1)).squeeze() #(batch_size, n_cand)
        scores_F = torch.matmul(cand, user_emb_F.unsqueeze(-1)).squeeze()
        _, ind_I = scores_I.topk(n_cand, dim=1) #(batch_size, n_cand)
        _, ind_F = scores_F.topk(n_cand, dim=1)
        pos_emb_I = torch.Tensor(batch_size, 2, self.dim_node).fill_(0).to(self.device) #(batch_size, k, dim_node)
        pos_emb_F = torch.Tensor(batch_size, 2, self.dim_node).fill_(0).to(self.device)
        neg_emb_I = torch.Tensor(batch_size, 2, self.dim_node).fill_(0).to(self.device)
        neg_emb_F = torch.Tensor(batch_size, 2, self.dim_node).fill_(0).to(self.device)
        for i in torch.arange(batch_size):
            pos_emb_I[i] = cand[i,ind_I[i][:2],:]
            pos_emb_F[i] = cand[i,ind_F[i][:2],:]
            neg_emb_I[i] = cand[i,ind_I[i][2:4],:]
            neg_emb_F[i] = cand[i,ind_F[i][2:4],:]
        loss_con = self.SSL_topk(user_emb_I, pos_emb_F, neg_emb_F)
        loss_con += self.SSL_topk(user_emb_F, pos_emb_I, neg_emb_I)        
        
        if self.module_test:
            var_input = ['users', 'candidates']
            var_inmed = ['x_parent', 'user4pos', 'pos_scores']
            var_ouput = ['loss', 'loss_reg']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )
        '''
        print(loss)
        print(loss_reg)
        print(loss_entropy)
        print(loss_con)
        '''
        return loss, loss_reg, loss_entropy, loss_con
    
    def SSL_topk(self, user, pos, neg):
        '''
        user: (batch_size, dim_node)
        pos:  (batch_size, k, dim_node)
        neg:  (batch_size, k, dim_node)
        '''
        user = F.normalize(user, p=2, dim=-1)
        pos  = F.normalize(pos,  p=2, dim=-1)
        neg  = F.normalize(neg,  p=2, dim=-1)
        pos_score = torch.matmul(pos, user.unsqueeze(-1)).squeeze() #(batch_size, k)
        neg_score = torch.matmul(neg, user.unsqueeze(-1)).squeeze()
        pos_score = torch.sum(torch.exp(pos_score / 0.2), 1)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), 1)
        con_loss = -torch.mean(torch.log(pos_score) - torch.log(pos_score + neg_score))
        return con_loss

    def bpr_loss_bak(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp: (batch_size, n_node)
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.pos: (batch_size, 1)
                candidates.neg: (batch_size, n_neg)
                candidates.stp: (batch_size, 1)
        output:
            loss, loss_reg
        '''
        x, pos, neg = self.prepareEmbFromID(self.embedding_item, users.history, candidates.pos, candidates.neg)
        adj = users.adj
        #x   : (batch_size, n_node, dim_node)
        #adj : (batch_size, n_node, n_node)
        #pos : (batch_size, 1     , dim_node)
        #neg : (batch_size, n_neg , dim_node)

        # add positional embedding
        x_PE = self.positionEncoder(users.timestp)
        x += x_PE 
        cdd_PE = self.positionEncoder(candidates.stp) # (batch_size, 1, dim_node)
        pos += cdd_PE
        n_neg = candidates.neg.shape[1]
        neg += cdd_PE.repeat(1,n_neg,1)

        # loss
        x_parent, s = self.parentNodeLayer(x, adj) #(batch_size, n_assign, dim_node), (batch_size, n_node, n_assign)
        
        user4pos = self.aggregateLayer(x_parent, pos) #(batch_size, 1, dim_node)
        pos_scores = self.predLayer(user4pos, pos) #(batch_size, 1)  
        
        neg_scores_list = []
        for i in range(n_neg):
            user4neg = self.aggregateLayer(x_parent, neg[:,i,:].unsqueeze(1)) #(batch_size, 1, dim_node)
            neg_scores_list.append(self.predLayer(user4neg, neg[:,i,:].unsqueeze(1))) #(batch_size, 1)
        neg_scores = torch.cat(neg_scores_list, dim=1) #(batch_size, n_neg)
        
        pos_scores = pos_scores.repeat(1,n_neg) #(batch_size, n_neg) 
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # entropy loss
        batch_size = s.size()[0]
        n_node = s.size()[1]
        loss_entropy = (-torch.sum(s*torch.log(s)))/(batch_size*n_node)

        # reg_loss
        batch_size = x.size()[0]
        loss_reg = (1/2)*(x.norm(2).pow(2)+
                        pos.norm(2).pow(2) +
                        neg.norm(2).pow(2))/float(batch_size)

        if self.module_test:
            var_input = ['users', 'candidates']
            var_inmed = ['x_parent', 'user4pos', 'pos_scores']
            var_ouput = ['loss', 'loss_reg']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )    
        return loss, loss_reg, loss_entropy

    def compute_rating(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.cdd: (batch_size, 1)
                candidates.stp: (batch_size, 1)
        output:
            cdd_scores: (batch_size)
        '''
        x, cdd = self.prepareEmbFromID(self.embedding_item, users.history, candidates.cdd)
        adj = users.adj

        # add positional embedding
        x_PE = self.positionEncoder(users.timestp)
        x += x_PE 

        cdd_PE = self.positionEncoder(candidates.stp)
        cdd += cdd_PE

        # loss
        x_parent, s = self.parentNodeLayer(x, adj) 
        #(batch_size, n_assign, dim_node) (batch_size, n_node, n_assign)
        
        user4cdd = self.aggregateLayer(x_parent, cdd) # (batch_size, 1, dim_node)
        cdd_scores = self.predLayer(user4cdd, cdd) # (batch_size, 1)

        cdd_scores = cdd_scores.squeeze()
        
        if self.module_test:
            var_input = ['users', 'candidates']
            var_inmed = ['x_parent', 'user4cdd']
            var_ouput = ['cdd_scores']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )
        
        return cdd_scores

    def compute_rating_byflatattention(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.cdd: (batch_size, 1)
                candidates.stp: (batch_size, 1)
        output:
            cdd_scores: (batch_size)
        '''
        x, cdd = self.prepareEmbFromID(self.embedding_item, users.history, candidates.cdd)
        adj = users.adj

        # add positional embedding
        x_PE = self.positionEncoder(users.timestp)
        x += x_PE 

        cdd_PE = self.positionEncoder(candidates.stp)
        cdd += cdd_PE

        # loss
        #x_parent, s = self.parentNodeLayer(x, adj) 
        #(batch_size, n_assign, dim_node) (batch_size, n_node, n_assign)
        
        user4cdd = self.aggregateLayer(x, cdd) # (batch_size, 1, dim_node)
        cdd_scores = self.predLayer(user4cdd, cdd) # (batch_size, 1)

        cdd_scores = cdd_scores.squeeze()
        
        return cdd_scores

    def compute_parent(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.cdd: (batch_size, 1)
                candidates.stp: (batch_size, 1)
        output:
            cdd_scores: (batch_size)
        '''
        x, cdd = self.prepareEmbFromID(self.embedding_item, users.history, candidates.cdd)
        adj = users.adj

        # add positional embedding
        x_PE = self.positionEncoder(users.timestp)
        x += x_PE 

        cdd_PE = self.positionEncoder(candidates.stp)
        cdd += cdd_PE #(batch_size, dim_node)

        # loss
        x_parent, s, z = self.parentNodeLayer.forward_casestudy(x, adj) 
        #(batch_size, n_assign, dim_node) (batch_size, n_node, n_assign)

        _weights = torch.mul(x_parent, cdd) # (batch_size, n_assign, dim_node)
        weights = torch.sum(_weights, dim=-1, keepdim=True).squeeze()[0] # (batch_size, n_assign, 1)

        weights_soft = torch.nn.Softmax(dim=0)(weights)

        return s[0].numpy(), weights_soft.numpy(), x_parent[0].numpy(), z[0].numpy(), x[0].numpy(), cdd[0].numpy() # (n_node, n_assign)

class Flat_Attention(nn.Module):
    def __init__(self, dim_node,\
                 num_items_all, \
                 module_test=False, **kwargs\
                 ):

        super(Flat_Attention, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.aggregateLayer  = AggregateLayer().to(device)
        self.predLayer       = PredLayer().to(device)       

        self.embedding_item = nn.Embedding(num_embeddings=num_items_all+1, embedding_dim=dim_node)
        if 1:#self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            #world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')

        self.module_test = module_test
    
    def prepareEmbFromID(self, Embedding, *ids):
        '''input:
                Embedding     : (n_item_all, dim_node)
                history_id    : (batch_size, n_node) 
                pos_id, neg_id: (batch_size, 1)
            output:
                x  : (batch_size, n_node, dim_node)
                pos: (batch_size, 1     , dim_node)
                neg: (batch_size, 1     , dim_node)
        '''
        return [Embedding(i) for i in ids]
    
    def bpr_loss(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.pos: (batch_size, 1)
                candidates.neg: (batch_size, 1)
        output:
            loss, loss_reg
        '''
        x, pos, neg = self.prepareEmbFromID(self.embedding_item, users.history, candidates.pos, candidates.neg)

        #x   : (batch_size, n_node, dim_node)
        #adj : (batch_size, n_node, n_node)
        #pos : (batch_size, 1     , dim_node)
        #neg : (batch_size, 1     , dim_node)

        # loss
        
        user4pos = self.aggregateLayer(x, pos) # (batch_size, 1, dim_node)
        pos_scores = self.predLayer(user4pos, pos)

        user4neg = self.aggregateLayer(x, neg) # (batch_size, 1, dim_node)
        neg_scores = self.predLayer(user4neg, neg)
        
        loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # reg_loss
        batch_size = x.size()[0]
        loss_reg = (1/2)*(x.norm(2).pow(2)+
                        pos.norm(2).pow(2) +
                        neg.norm(2).pow(2))/float(batch_size)
        
        if self.module_test:
            var_input = ['users', 'candidates']
            var_inmed = ['user4pos', 'pos_scores']
            var_ouput = ['loss', 'loss_reg']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )
        
        return loss, loss_reg

    def compute_rating(self, users, candidates):
        '''
        input:
            users: namedtuple
                users.history: (batch_size, n_node)
                users.timestp
                users.adj    : (batch_size, n_node, n_node)
            candidates: namedtuple
                candidates.cdd: (batch_size, 1)
                candidates.stp: (batch_size, 1)
        output:
            cdd_scores: (batch_size)
        '''
        x, cdd = self.prepareEmbFromID(self.embedding_item, users.history, candidates.cdd)

        # loss        
        user4cdd = self.aggregateLayer(x, cdd) # (batch_size, 1, dim_node)
        cdd_scores = self.predLayer(user4cdd, cdd) # (batch_size, 1)

        cdd_scores = cdd_scores.squeeze()
        
        if self.module_test:
            var_input = ['users', 'candidates']
            var_inmed = ['x_parent', 'user4cdd']
            var_ouput = ['cdd_scores']
            locals_cap = locals()
            module_test_print(
                dict(zip(var_input, [eval(v, locals_cap) for v in var_input])),\
                dict(zip(var_inmed, [eval(v, locals_cap) for v in var_inmed])),\
                dict(zip(var_ouput, [eval(v, locals_cap) for v in var_ouput]))
                )
        
        return cdd_scores

if __name__=='__main__':

    from utils import module_test_print

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Module_Test = DUVRec(dim_node=3, dim_hidden=3, n_assign=2, n_heads=2, num_items_all=10, module_test=True).to(device)
    
    '''
    from collections import namedtuple
    Users = namedtuple('User', ['history', 'adj'])
    Candidates = namedtuple('Candidates', ['pos', 'neg'])

    history = torch.tensor([[1,2],[3,4]])
    adj = torch.tensor([[1.,1.],[0.,1.]]).repeat(2,1).view((2,2,2))
    pos = torch.tensor([[5],[6]])
    neg = torch.tensor([[7],[8]])

    users = Users._make([history, adj])
    candidates = Candidates._make([pos, neg])

    output = Module_Test.bpr_loss(users, candidates)
    '''

    from collections import namedtuple
    Users = namedtuple('User', ['history', 'adj', 'timestp'])
    Candidates = namedtuple('Candidates', ['cdd', 'stp'])

    history = torch.tensor([[1,2],[1,2]]).to(device)
    timestp = torch.tensor([[11,21],[11,21]]).to(device)
    adj = torch.tensor([[0.,1.],[1.,0.]]).repeat(2,1).view((2,2,2)).to(device)
    cdd = torch.tensor([[5],[6]]).to(device)
    stp = torch.tensor([[51],[61]]).to(device)

    users = Users._make([history, adj, timestp])
    candidates = Candidates._make([cdd, stp])

    output = Module_Test.compute_rating(users, candidates)

