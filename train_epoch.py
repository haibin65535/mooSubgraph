import torch
from utils import num_graphs
import numpy as np
from min_norm_solvers import MinNormSolver
import torch.nn.functional as F 
import torch.nn as nn
from torch.autograd import Variable
import time
import math


def gaussian_kl(edge_att,split_e):
    edge_atts = [i.T for i in list(torch.split(edge_att,list(split_e.detach().cpu().numpy())))] 
    losses = []
    for G_edge_att in edge_atts:
        G_edge_att = torch.log_softmax(G_edge_att,dim=1)
        gaussian = torch.softmax(torch.randn_like(G_edge_att),dim = 1)
        losses.append(F.kl_div(G_edge_att,gaussian,reduction='sum'))
    return sum(losses)/len(losses)

def static_weight_loss(model, optimizer, loader, device,args):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    total_loss_s = 0
    total_loss_m = 0
    total_loss_ge = 0
    correct = 0

    for it, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)
        un_one_hot_target = data.uny/(model.num_classes - 1)
        
        xs_logs,xm_logs,edge_att,node_att = model(data)
        
        G_att_loss =  gaussian_kl(edge_att,data.split_e)*0.01
        s_loss = F.kl_div(xs_logs, un_one_hot_target, reduction='batchmean')*0.1#-(un_one_hot_target * xs_logs).sum(dim=-1).mean()
        m_loss = F.nll_loss(xm_logs, one_hot_target) 
        loss =  args.s*s_loss + args.m*m_loss + args.ge*G_att_loss
        
        start = time.time()
        loss.backward()
        mytime = (time.time() - start)*1000
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        pred = xm_logs.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_s += s_loss.item() * num_graphs(data)
        total_loss_m += m_loss.item() * num_graphs(data)
        total_loss_ge += G_att_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(loader.dataset)
    losses = {
        'total' :total_loss / num,
        's':total_loss_s / num,
        'm':total_loss_m / num,
        'ge':total_loss_ge / num
    }
    correct = correct / num
    return mytime,losses,correct

def get_parameters_grad(model):
    grads = []
    for param in model.Head_encoder.parameters():
        if param.grad is not None:
            grads.append(Variable(param.grad.data.clone(), requires_grad=False))
    return grads

def mgda_loss(model, optimizer, loader, device,args):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    total_loss_s = 0
    total_loss_m = 0
    total_loss_ge = 0

    correct = 0
    saved_loss = {}
    mgda = {
        'ge':[],
        's':[],
        'm':[]
    }
    loss_name = mgda.keys()
    for it, data in enumerate(loader):
        
        optimizer.zero_grad()
        data = data.to(device)    
        one_hot_target = data.y.view(-1)
        un_one_hot_target = data.uny/(model.num_classes - 1)

        xs_logs,xm_logs,edge_att,node_att = model(data)
        
        G_att_loss =  gaussian_kl(edge_att,data.split_e)*0.01
        s_loss = F.kl_div(xs_logs, un_one_hot_target, reduction='batchmean')*0.1
        m_loss = F.nll_loss(xm_logs, one_hot_target) 
        
        loss_data = {}
        grads = {}
        start = time.time()

        #-----------------------------mgda--------------------------------
        loss_data['ge'] = G_att_loss.data
        saved_loss['ge'] = G_att_loss
        
        G_att_loss.backward(retain_graph=True)
        grads['ge'] = get_parameters_grad(model)
        model.zero_grad()
        
        #-----------------------------------------------------------------
        loss_data['s'] = s_loss.data
        saved_loss['s'] = s_loss

        s_loss.backward(retain_graph=True)
        grads['s'] = get_parameters_grad(model)
        model.zero_grad()

        #-------------------------------------------------------------
        loss_data['m'] = m_loss.data
        saved_loss['m'] = m_loss

        m_loss.backward(retain_graph=True)
        grads['m'] = get_parameters_grad(model)
        model.zero_grad()
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        
        gn = MinNormSolver.gradient_normalizers(grads, loss_data, 'l2')
    
        for t in loss_data:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t].to(grads[t][gr_i].device)
        sol, _ = MinNormSolver.find_min_norm_element_FW([grads[t] for t in loss_name])
        sol = {k:sol[i] for i, k in enumerate(loss_name)}
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        loss = []
        for  key in  loss_name:
            loss.append(saved_loss[key]*sol[key])
            mgda[key].append(sol[key])
        loss = sum(loss)
        loss.backward()
        mytime = (time.time() - start)*1000

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        pred = xm_logs.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        total_loss += loss.item() * num_graphs(data)
        total_loss_s += s_loss.item() * num_graphs(data)
        total_loss_m += m_loss.item() * num_graphs(data)
        total_loss_ge += G_att_loss.item() * num_graphs(data)
        optimizer.step()
    
    for  key in  loss_name:
        setattr(args,key,sum(mgda[key])/len(mgda[key]))

    num = len(loader.dataset)
    losses = {
        'total' :total_loss / num,
        's':total_loss_s / num,
        'm':total_loss_m / num,
        'ge':total_loss_ge / num
    }

    correct = correct / num
    return mytime,losses, correct

# handle for flexible external changes to the training mode
funcs = {
    'swl':static_weight_loss,
    'mgda':mgda_loss,
}

