import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import tensor
import numpy as np
from min_norm_solvers import MinNormSolver
from utils import k_fold
import time
from train_epoch import funcs
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_savefile(args,save_model):
    savefile = 'saved/{}/'.format(save_model) + args.dataset+ '/' 
    if not os.path.exists(savefile):
        os.mkdir(savefile)

    return savefile + args.model
    
def savefile(args,save_model,data):
    savefile = 'saved/{}/'.format(save_model) + args.dataset + '/'
    
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    if isinstance(data,dict):
        np.save(savefile + args.model,data)
    np.save(savefile + args.model,np.array(data))

def train_causal_dir(train_set, val_set,  test_set, model_func=None, args=None):
    train_causal_epoch = funcs[args.train_model]

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)

    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
   
    best_accs_folds = []
    mgda_s = [] 
    for fold in range(args.folds):
 
        model = model_func(args.feature_dim, args.num_classes).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
        times = []
        names = ['Train','val','Test']
        best_accs = [0. for _ in names]
        best_epoch = [0 for _ in names]
        stop_updates = [0 for _ in names]
        for epoch in range(1, args.epochs + 1):
            mytime,losses, train_acc = train_causal_epoch(model, optimizer, train_loader, device,args)
            lr_scheduler.step()
            times.append(mytime)
            val_acc, _, _ = eval_acc_causal(model, val_loader, device)
            test_acc, _, _ = eval_acc_causal(model, test_loader, device)
            accs = [train_acc,val_acc,test_acc]

            if best_accs[2] < accs[2]:
                best_accs = accs
                best_epoch[2] = epoch
                stop_updates[2] = 0
            else:
                stop_updates[2] += 1 
    
            info_out_print = "flod:[{}] | dataset:[{}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f} = S({:.4f}*{:.4f}) + M({:.4f}*{:.4f}) + ge({:.4f}*{:.4f})] \n".format(
                             fold,args.dataset,args.model,epoch, args.epochs,losses['total'], args.s, losses['s'], args.m,losses['m'],args.ge,losses['ge'])
            acc_out = "\tACC:"
            
            best_acc_out = "\n\tBESTACC:"
            for i in  range(len(names)):
                acc_out = acc_out + names[i] + ": [{:.2f}]".format(accs[i]*100)
                best_acc_out = best_acc_out + names[i] + ": [{:.2f}] Update: [{}]".format(best_accs[i]*100,best_epoch[i])
    
            #if stop_updates[2]>40:
                #info_out_print = 'early stop: | ' + info_out_print
                #break
            print(info_out_print + acc_out + best_acc_out + "\n use time: [{:2f}]".format(mytime) + '\n') 
        
        print(info_out_print + best_acc_out + "\n epoch use time: [{:2f}]".format(sum(times)/len(times)))
        best_accs_folds.append(best_accs)


    best_accs_folds = np.array(best_accs_folds)[:,2]
    mean = best_accs_folds.mean()*100
    std = best_accs_folds.std()*100
    print('test:[{:.2f} ± {:.2f}]'.format(mean,std))
    return mean,std
    
def train_ms_dir(train_set, val_set,  test_set, model_func=None, args=None):
    train_causal_epoch = funcs[args.train_model]

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False)
    
    if args.feature_dim == -1:
        args.feature_dim = args.max_degree
   
    model = model_func(args.feature_dim, args.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
    
    edge_atts,node_atts,times = [],[],[]
    names = ['Train','val','Test']
    best_accs = [0. for _ in names]
    best_epoch = [0 for _ in names]
    losses = []
    acces = []
    outs = {}
    for epoch in range(1, args.epochs + 1):
        mytime,losses, train_acc = train_causal_epoch(model, optimizer, train_loader, device,args)
        lr_scheduler.step()
        times.append(mytime)
        losses.append(np.array(list(losses.values())))
        val_acc,edge_att,node_att,out =  eac_moreinfo(model, val_loader, device)
        test_acc,_,_ = eval_acc_causal(model, test_loader, device)
        accs = [train_acc,val_acc,test_acc]
        acces.append(accs)
        if best_accs[2] < accs[2]:
            best_accs = accs
            best_epoch[2] = epoch
        
        edge_atts.append(edge_att.detach().cpu().numpy())
        node_atts.append(node_att.detach().cpu().numpy())
        outs[epoch] = out

        info_out_print =  "dataset:[{}] | Model:[{}] Epoch:[{}/{}] Loss:[{:.4f} = S({:.4f}*{:.4f}) + M({:.4f}*{:.4f}) + ge({:.4f}*{:.4f})] \n".format(
                             args.dataset,args.model,epoch, args.epochs,losses['total'], args.s, losses['s'], args.m,losses['m'],args.ge,losses['ge'])
        acc_out = "\tACC:"
        best_acc_out = "\n\tBESTACC:"
        for i in  range(len(names)):
            acc_out = acc_out + names[i] + ": [{:.2f}]".format(accs[i]*100)
            best_acc_out = best_acc_out + names[i] + ": [{:.2f}] Update: [{}]".format(best_accs[i]*100,best_epoch[i])
    
        print(info_out_print + acc_out + best_acc_out + "\n use time: [{:2f}]".format(mytime) + '\n') 
        
    print(info_out_print + best_acc_out + "\n epoch use time: [{:2f}]".format(sum(times)/len(times)))

    savefile(args,'info',outs) 
    savefile(args,'edge_atts',edge_atts)
    savefile(args,'node_atts',node_atts)
    savefile(args,'ACC',acces)
    savefile(args,'loss',losses)
    torch.save(model, get_savefile(args,'model') + '.pt') 

def train_causal_real(dataset=None, model_func=None, args=None):
    train_accs, test_accs= [], []
    best_epochs = []
    times = []
    losses_saved =  []
    train_causal_epoch = funcs[args.train_model]

    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, args.folds, args.epoch_select))):
    
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx.long()]
        
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
        
        model = model_func(dataset.num_features, dataset.num_classes).to(device)
        
        losses = []
        best_test_acc, best_epoch = 0, 0
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=-1, verbose=False)
        #from warmup_scheduler import GradualWarmupScheduler
        #warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier = 1, total_epoch=args.epochs, after_scheduler=lr_scheduler)
        for epoch in range(1, args.epochs + 1): 
            mytime,losses, train_acc = train_causal_epoch(model, optimizer, train_loader, device,args)
            #lr_scheduler.step()
            #warmup_scheduler.step()
            test_acc,edge_atts,node_atts = eval_acc_causal(model, test_loader, device)
            losses.append(np.array(list(losses.values())))
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                
        
            times.append(mytime)
            best_epochs.append(best_test_acc)
            print("Causal | dataset:[{}] fold:[{}] | Epoch:[{}/{}] Loss:[{:.4f} = S({:.4f} * {:.4f}) + M({:.4f} * {:.4f}) + ge({:.4f} * {:.4f})  ] Train:[{:.4f}] Test:[{:.2f}] | Best Test:[{:.2f}] at Epoch:[{}] | use time : {:.6f}"
                    .format(args.dataset,
                            fold,
                            epoch, args.epochs,
                            losses['total'], args.s, losses['s'], args.m,losses['m'],args.ge,losses['ge'],
                            train_acc * 100,  
                            test_acc * 100, 
                            best_test_acc * 100, 
                            best_epoch,
                            mytime))
        
        losses_saved.append(losses)
        print("syd: Causal fold:[{}] | Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] at epoch [{}]"
                .format(fold,
                        args.dataset,
                        args.model,
                        best_test_acc * 100, 
                        best_epoch,
                        ))
    

    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    train_acc = train_acc.view(args.folds, args.epochs)
    test_acc = test_acc.view(args.folds, args.epochs)
    
    _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
    selected_epoch = selected_epoch.repeat(args.folds)
    
    test_acc = test_acc[torch.arange(args.folds, dtype=torch.long), selected_epoch]

    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    best_epochs_mean = sum(best_epochs)/args.folds

    print("=" * 150)
    print('sydall Final: Causal | Dataset:[{}] Model:[{}] seed:[{}]| best_epoch = {:.2f} | Test Acc: {:.2f}±{:.2f} | [Settings] s:{},m:{},harf:{},dim:{},fc:{} | epoch use time :{:.6f}'
         .format(args.dataset,
                 args.model,
                 args.seed,
                 best_epochs_mean,
                 test_acc_mean * 100, 
                 test_acc_std * 100,
                 args.s,
                 args.m,
                 args.harf_hidden,
                 args.hidden,
                 args.fc_num,
                 sum(times)/len(times)))
    print("=" * 150)
    return test_acc_mean,test_acc_std
    

def eval_acc_causal(model, loader, device):
    model.eval()
    correct_m = 0
    correct_s = 0
    edge_atts,node_atts = [],[]
    for i,data in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            s_logs, m_logs, edge_att,node_att = model(data)
            edge_atts.append(edge_att)
            node_atts.append(node_att)
            pred_m = m_logs.max(1)[1]
            pred_s = s_logs.min(1)[1]
        correct_m += pred_m.eq(data.y.view(-1)).sum().item()
        correct_s += pred_s.eq(data.y.view(-1)).sum().item()

    acc = correct_m  / len(loader.dataset)
    edge_atts = torch.cat(edge_atts)
    node_atts = torch.cat(node_atts)
    return acc,edge_atts,node_atts


def eac_moreinfo(model, loader, device):
    model.eval()
    correct_m = 0
    correct_s = 0
    edge_atts,node_atts,outs = [],[],[]
     
    for i,data in enumerate(loader):
        data = data.to(device)
        with torch.no_grad():
            s_logs, m_logs, edge_att,node_att,out = model(data, True)
            edge_atts.append(edge_att)
            node_atts.append(node_att)
            outs.append(out)
            pred_m = m_logs.max(1)[1]
            pred_s = s_logs.min(1)[1]
        correct_m += pred_m.eq(data.y.view(-1)).sum().item()
        correct_s += pred_s.eq(data.y.view(-1)).sum().item()

    acc = correct_m  / len(loader.dataset)
    edge_atts = torch.cat(edge_atts)
    node_atts = torch.cat(node_atts)
    return acc,edge_atts,node_atts,outs