from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv

class Head_encoder(torch.nn.Module):
    def __init__(self , num_features,
                        args,
                        bns_conv,
                        convs):
        super(Head_encoder, self).__init__()
        self.args = args
        self.noise_std = args.noise
        hidden_in = num_features
        hidden = args.hidden

        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = bns_conv
        self.convs = convs
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self,x,edge_index,use_bns_conv = True):
        if self.training:
            x = x + torch.randn_like(x) * self.noise_std
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        if use_bns_conv :
            for i, conv in enumerate(self.convs):
                x = self.bns_conv[i](x)
                x = F.relu(conv(x, edge_index))
        else:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                
        return x

def lazy_soft(x,split_):
    batch_size = split_.shape[0]
    x = list(torch.split(x,list(split_.detach().cpu().numpy())))
    for i in range(batch_size):
        x[i] = torch.softmax(x[i],dim = 0)
    return torch.cat(x)
    
class Att_cov(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden,
                gfn=False,edge_norm=True):
        super(Att_cov, self).__init__()
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        self.edge_att_mlp = nn.Linear(in_channels*2,1)       
        self.node_att_cov = GConv(in_channels,1)
        
    def forward(self,x,edge_index,split_n):
        row, col = edge_index        #  2*5016
        edge_rep = torch.cat([x[row], x[col]], dim=-1) #5016*c*2
        edge_att = self.edge_att_mlp(edge_rep)  #5016,     
        edge_att = F.sigmoid(edge_att) 
        edge_att_m = edge_att
        edge_att_s = 1 - edge_att
        
        node_att = self.node_att_cov(x,edge_index)
        node_att_m = lazy_soft(node_att,split_n)
        node_att_s = lazy_soft(1 - node_att_m,split_n) 
        return edge_att_m,edge_att_s,node_att_m,node_att_s

class Split_sub_encoder(torch.nn.Module):
    def __init__(self, args,gfn=False,edge_norm=True):
        super(Split_sub_encoder, self).__init__()
        self.args = args
        hidden = args.hidden
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn,drop_out = args.drop_out)
        self.bn = BatchNorm1d(hidden)
        self.conv = GConv(hidden, hidden)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
    
    def forward(self, x ,edge_index, batch,edge_att,node_att):
        x = node_att * x
        x = F.relu(self.conv(self.bn(x), edge_index,edge_att))
        x = self.global_pool(x, batch)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, num_classes, args):
        super(Classifier, self).__init__()

        self.args = args
        
        hidden = args.hidden
        hidden_out = num_classes
        
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self,x):
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class MS_connet(torch.nn.Module):
    def __init__(self,args):
        super(MS_connet, self).__init__()
        hidden = args.hidden
        self.fc = Linear(hidden*2,hidden)

    def forward(self,xs,xm):
        att = self.fc(torch.cat([xs,xm],dim = 1))
        att = torch.tanh(att) #torch.where(att > 0,1,0)
        xs = xs*att
        return xm + xs


class Template(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes, args,
                       gfn=False, 
                       header_mdoe = 'GCN',
                       head=4,
                       edge_norm=True):
        super(Template, self).__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.num_classes = num_classes
        
        self.use_bns_conv = True
        bns_conv = torch.nn.ModuleList()
        convs = torch.nn.ModuleList()
        
        if header_mdoe == 'GCN':
            for i in range(num_conv_layers):
                GConv = partial(GCNConv,edge_norm=edge_norm, gfn=gfn,drop_out = args.drop_out)
                bns_conv.append(BatchNorm1d(hidden))
                convs.append(GConv(hidden, hidden))
        
        elif header_mdoe == 'GIN':
            self.use_bns_conv = False
            for i in range(num_conv_layers):
                convs.append(GINConv(
                Sequential(
                        Linear(hidden, hidden), 
                        BatchNorm1d(hidden), 
                        ReLU(),
                        Linear(hidden, hidden), 
                        ReLU(),
                        nn.Dropout(args.drop_out)
                )))
        
        elif header_mdoe == 'GAT':
            for i in range(num_conv_layers):
                bns_conv.append(BatchNorm1d(hidden))
                convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=args.drop_out))

        self.Head_encoder = Head_encoder(num_features,args,bns_conv,convs)
        self.att_cov = Att_cov(hidden,hidden)
        self.Separate_sub_encoder = Split_sub_encoder(args,gfn,edge_norm)
        self.Major_sub_encoder = Split_sub_encoder(args,gfn,edge_norm)
        self.Split_Classifier = Classifier(num_classes,args)
        self.MS_connet = MS_connet(args)
        self.Major_Classifier = Classifier(num_classes,args)

    def forward(self, data,dict_out = False):
        out = {}
        x,edge_index,batch,split_n, = data.x,data.edge_index, data.batch,data.split_n
        x = self.Head_encoder(x,edge_index,self.use_bns_conv)
        out['head'] = global_mean_pool(x,batch)
        edge_att_m,edge_att_s,node_att_m,node_att_s = self.att_cov(x,edge_index,split_n)
        xm = self.Major_sub_encoder(x,edge_index,batch,edge_att_m,node_att_m)
        out['xm'] = xm
        xs = self.Separate_sub_encoder(x,edge_index,batch,edge_att_s,node_att_s)
        out['xs'] = xs
        xm = self.MS_connet(xs , xm)
        out['xms'] = xm
        xs_logs = self.Split_Classifier(xs)
        xm_logs = self.Major_Classifier(xm)
        
        if dict_out:
            return xs_logs,xm_logs,edge_att_m,node_att_m,out
        
        return xs_logs,xm_logs,edge_att_s,node_att_s
        #  nl_loss(xs_logs ,  1-y)
        #  nl_loss(xm_logs ,  y)

class GCNNet(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, hidden, 
                       num_feat_layers=1, 
                       num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0.2, 
                 edge_norm=True):
        super(GCNNet, self).__init__()

        self.global_pool = global_mean_pool
        self.dropout = nn.Dropout(dropout)
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
            
        x = self.global_pool(x, batch)
        x = self.dropout(x)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))
        
        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)

class GINNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes,
                       hidden, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU()
                      )))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        # x, edge_index, batch = data.feat, data.edge_index, data.batch
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))    
        x = self.bn_hidden(x)
        x = self.lin_class(x)

        prediction = F.log_softmax(x, dim=-1)
        return prediction

class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_classes,
                       hidden,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)