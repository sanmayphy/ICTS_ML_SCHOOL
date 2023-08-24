# ICTS_ML_SCHOOL

## Setting up the environments

Install MiniConda using the instructions at [Conda Website](https://docs.conda.io/en/latest/miniconda.html#quick-command-line-install)

Create a working environment 

```
conda create --name work_env
conda activate work_env
```
Pytorch installation following [PyTorch Website](https://pytorch.org/get-started/locally/)

```
pip install torch torchvision torchaudio
```
Pytorch-Geometric installation following [PyG Website](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

```
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```
For MAC arm64 users, please use the following command 
```
MACOSX_DEPLOYMENT_TARGET=13.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+${cpu}.html
MACOSX_DEPLOYMENT_TARGET=13.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${cpu}.html
MACOSX_DEPLOYMENT_TARGET=13.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+${cpu}.html
MACOSX_DEPLOYMENT_TARGET=13.5 CC=clang CXX=clang++ python -m pip --no-cache-dir  install  torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+${cpu}.html
MACOSX_DEPLOYMENT_TARGET=13.5 CC=clang CXX=clang++ python -m pip --no-cache-dir install git+https://github.com/pyg-team/pyg-lib.git
```


Install uproot and awkward array for reading ROOT files [uproot](https://uproot.readthedocs.io/en/latest/index.html)
```
pip install uproot awkward
```
