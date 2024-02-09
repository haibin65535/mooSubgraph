from datasets import get_dir_dataset
import torch
from train import train_baseline_dir
from train_case import train_Case_dir
import opts
import os
import warnings
import numpy as np
from functools import partial
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
warnings.filterwarnings('ignore')

def train_grid(config,args):
    args.s = config['s']
    args.m = config['m']
    train_dataset, val_dataset,test_dataset = get_dir_dataset(args)
    model_func = opts.get_model(args)
    if args.model in ["GIN","GCN", "GAT"]:
        trainfun = partial(train_baseline_dir,train_dataset, val_dataset, test_dataset,model_func=model_func)
    elif args.model in ["CaseGCN", "CaseGIN", "CaseGAT"]:
        trainfun = partial(train_Case_dir,train_dataset, val_dataset, test_dataset,model_func=model_func)
    else:
        assert False

    test_acc_mean,test_acc_std =  trainfun(args=args)
    return {'mean_accuracy':test_acc_mean
            ,'std':test_acc_std}

def main():
    args = opts.parse_args()
    args.hidden = 32
    args.data_root = '~/wenhaibin/CAL/CAL-sp/'
    

    if args.tune:
        sched = ASHAScheduler()
        config = {
            's': tune.grid_search(np.linspace(0,1,11)),
            'm': tune.grid_search(np.linspace(0,1,11)),
        }
        analysis = tune.run(
            partial(train_grid,args = args),
            metric="mean_accuracy",
            mode="max",
            name= args.dataset + '/'+ args.model + '_grid',
            scheduler=sched,
            resources_per_trial={"cpu": 2,"gpu": 1},
            num_samples = 1,
            config = config)
        
        df = analysis.dataframe()
        savefile = 'saved/tune/grid/'+ args.dataset +'/' + args.model + '/'
        if not os.path.exists(savefile):
            os.mkdir(savefile)
        df.to_csv(savefile + "tune_data.csv")
    else:
        train_dataset, val_dataset,test_dataset = get_dir_dataset(args)
        model_func = opts.get_model(args)
        if args.model in ["GIN","GCN", "GAT"]:
            trainfun = partial(train_baseline_dir,train_dataset, val_dataset, test_dataset,model_func=model_func)
        elif args.model in ["CaseGCN", "CaseGIN", "CaseGAT"]:
            trainfun = partial(train_Case_dir,train_dataset, val_dataset, test_dataset,model_func=model_func)
        else:
            assert False
        trainfun(args=args)     
    
if __name__ == '__main__':
    main()