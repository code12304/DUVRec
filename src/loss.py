
import torch
from torch import nn, optim
import config

class BPRLoss:
    def __init__(self, 
                 recmodel, 
                 decay_reg, decay_ent, decay_con, lr, **kwargs):
        self.model = recmodel
        self.decay_reg = decay_reg
        self.decay_ent = decay_ent
        self.decay_con = decay_con
        self.lr = lr
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        
    def stageOne(self, users, candidates):
        
        if config.model_choice=='DUVRec':
            
            loss, reg_loss, entropy_loss, con_loss = self.model.bpr_loss(users, candidates)
            '''
            print('*'*10)
            print(loss)
            print(reg_loss)
            print(entropy_loss)
            print(con_loss)
            print('*'*10)
            '''
            reg_loss = reg_loss*self.decay_reg
            entropy_loss = entropy_loss*self.decay_ent
            con_loss = con_loss*self.decay_con
            loss = loss + reg_loss + entropy_loss + con_loss

        elif config.model_choice=='flat_attention':
            
            loss, reg_loss = self.model.bpr_loss(users, candidates)
            reg_loss = reg_loss*self.decay_reg
            loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()
