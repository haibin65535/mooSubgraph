## Dependencies

Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).
Typically, you might need to run the following commands:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric == 2.0.2
pip install torch-scatter  == 2.0.9
pip install torch-sparse == 0.6.15 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install networkx                    
pip install matplotlib  
pip install dgl-cu101
```

## Experiments

### For dir datasets 
```
python main_dir.py --model 'CaseGCN' --dataset Spurious-Motif --bias 0.5
python main_dir.py --model 'CaseGCN' --dataset Spurious-Motif --bias 0.5 --train_model mgda
```
### For TU datasets

```
python main_real.py --model CaseGAT --dataset MUTAG  
python main_real.py --model CaseGAT --dataset MUTAG --train_model mgda
```

## Data download
dir datasets can be downloaded [here](https://github.com/Wuyxin/DIR-GNN).
TU datasets can be downloaded when you run ``main_real.py``.

