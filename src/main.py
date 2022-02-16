

import utils
import torch
import numpy as np
import random
import time
import os
import procedure

import config
print(config.loss)

import loss
import model
model_class = {
    'DUVRec'       : model.DUVRec,
    'flat_attention': model.Flat_Attention
}

print(f"dataset: {config.dataset_choice}")
print(f"model  : {config.model_choice}")
print(f"device : {config.info['device']}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(63)
print('random seed set...')

Recmodel = model_class[config.model_choice](
                **config.model_args[config.model_choice]
            )\
           .to(config.info['device'])
bpr = loss.BPRLoss(recmodel=Recmodel, **config.loss)

k = config.test['k']
best_epoch = {f'mrr@{k}':-float('inf')}
log = open(config.info['weight_folder']+'log_'+f"{config.model_choice}_{config.dataset_choice}.txt", 'w')
acc_trace = utils.AccTrace()

for epoch in range(config.train['num_epoch']):
    print('======================')
    print(f'EPOCH[{epoch}]')
    start = time.time()

    output_information = procedure.BPR_train(recmodel=Recmodel, loss_class=bpr, **config.train)

    if epoch % config.train['tr_per_te'] == 0:
        utils.cprint("[TEST]")
        best_epoch
        result = procedure.Test(recmodel=Recmodel, epoch=epoch, **config.test) # {'prec': prec, 'rr':rr}
        print("#temp BEST:", best_epoch)

        torch.save(Recmodel.state_dict(), \
            config.info['weight_folder']+f"{config.model_choice}_{config.dataset_choice}_{epoch}.pth.tar")
        log.write(f"epoch:{epoch:5d} "+str(result)+'\n')

        if result[f'mrr@{k}'] < best_epoch[f'mrr@{k}']:
            '''累计n次小于最优值就early stop'''
            acc_trace.fail_add_one()
            if acc_trace.count == config.train['num_earlyStop']:
                utils.cprint("[EARLY STOP]")
                print('*'*15)
                print("BEST EPOCH:\n", best_epoch)
                print('*'*15)
                break
        else:
            best_epoch['epoch'] = epoch
            best_epoch.update(result)
            #if epoch!=0:
            #    os.remove(utils.get_weight_file(config.model_choice, epoch-config.train['tr_per_te']))
            acc_trace.reset()

    #output_information = procedure.BPR_train(recmodel=Recmodel, loss_class=bpr, **config.train)
    
    print(f'{output_information}')
    '''
    print(f'[saved][{output_information}]')
    torch.save(Recmodel.state_dict(), config['weight_file'])
    print(f"[TOTAL TIME] {time.time() - start}")
    '''
log.close()