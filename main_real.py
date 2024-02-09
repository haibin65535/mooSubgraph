from datasets import get_dataset
from train_causal import train_causal_real
import opts
import warnings
warnings.filterwarnings('ignore')
from train import train_baseline_real 
import numpy as np

from functools import partial
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_grid(config,args):
    args.s = config['s']
    args.m = config['m']
    args.data_root = '~/wenhaibin/CAL/CAL-sp/data'
    dataset_name, feat_str, _ = opts.create_n_filter_triples([args.dataset])[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
    
    model_func = opts.get_model(args)
    if args.model in ["GIN","GCN", "GAT"]:
        trainfun = partial(train_baseline_real,dataset, model_func)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        trainfun = partial(train_causal_real,dataset=dataset, model_func=model_func)
    else:
        assert False
    test_acc_mean,test_acc_std =  trainfun(args=args)

    return {'mean_accuracy':test_acc_mean
            ,'std':test_acc_std}


def main():
    args = opts.parse_args()
    args.noise = 0
    dataset_name, feat_str, _ = opts.create_n_filter_triples([args.dataset])[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
    print(dataset)
    model_func = opts.get_model(args)
    if args.model in ["GIN","GCN", "GAT"]:
        trainfun = partial(train_baseline_real,dataset, model_func)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        trainfun = partial(train_causal_real,dataset=dataset, model_func=model_func)
    else:
        assert False
    
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
        df.to_csv('saved/tune/grid/'+ dataset_name +'/' + args.model + "tune_data.csv")
    else:
        trainfun(args=args)     
if __name__ == '__main__':
    main()