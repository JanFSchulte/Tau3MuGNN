# Tau3MuGNNs
This repo is to reproduce the results of Tau3MuGNNs project. Created by Siqi Miao (Georgia Tech) and updated by Benjamin Simon (Purdue).

# Use Gilbreth
## 0. connect VPN
To ssh to the Gilbreth cluster, one should first establish a VPN connection to `webvpn.purdue.edu`. Tutorials on how to do this can be found [here](https://www.itap.purdue.edu/newsroom/2020/200316-webvpn.html).

Basically, you need to first install `Cisco AnyConnect` and then connect to `webvpn.purdue.edu`, where the username is `purdue_id` (e.g. `miao61`) and the password is `[pin],push`. The `[pin]` is the 4-digit PIN for [BoilerKey Two-Factor Authentication](https://www.purdue.edu/apps/account/BoilerKey/).

## 1. ssh to Gilbreth
```
ssh purdue_id@gilbreth.rcac.purdue.edu
```
Then, it will ask for a password, which is also `[pin],push` for the BoilerKey.

## 2. change directory
I was recommended to use the `scratch` directory instead of the default home directory.
```
cd /scratch/gilbreth/purdue_id/
```

## 3. setup ssh key for GitHub
Since our repo is private, we need to setup a ssh key for GitHub. [Here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) provides a tutorial on how to do this.

Then, clone the repo:
```
git clone git@github.com:simon71701/Tau3MuGNNs.git
cd Tau3MuGNNs
```


## 4. install anaconda
Run:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
sh Anaconda3-2021.11-Linux-x86_64.sh
```
It will ask where to install, rememeber install it at `/scratch/gilbreth/purdue_id/anaconda3`. It may take a while to install. Type `yes` for all options. After installation, activate `conda` command by `source ~/.bashrc` everythime logging the server.

Now we should be at the `base` environment. Type `python`, we shall see a version of `Python 3.9.7`. Then type `quit()` to quit python, and start installing packages.

## 5. create conda environment and install packages

First, create a conda environment and activate it.
```
conda create --p ./tau3mu python=3.9
conda activate tau3mu
```

Install dependencies:

```
conda install -y pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.9.0+cu102.html
pip install -r requirements.txt
```

# Get the data
To run the code, we need to put those `.pkl` files in the right place.

When running the code, those dataframes will be processed according to the specific setting, and the processed files will be saved under `$ProjectDir/data/processed-[setting]-[cut_id]`. In this project, for simplicity we call them `SignalPU0, SignalPU200, BkgPU200` as `pos0, pos200, neg200` respectively.

Please note that the processed files may take up lots of disk space (5 gigabytes+), and when processing them it may also take up lots of memory (10 gigabytes+).

# Train a model

We provide `8` settings for training GNNs, and the corresponding configurations can be found in `$ProjectDir/src/configs/`. To train a GNN with a specific setting, one can do:

```
cd ./src
python train_gnn.py --setting [setting_name] --cuda [GPU_id] --cut [cut_id]
```

`GPU_id` is the id of the GPU to use. To use CPU, please set it to `-1`. `cut_id` is the id of the cut to use. Its default value is `None` and can be set to `cut1` or `cut1+2`. Note that when some cut is used, the `pos_neg_ratio` may need to be adjusted because many positive samples will be dropped.


One thing to notice is that if you have had processed files for a specific setting, even then you change some data-options in the config file, the processed files will not be changed in the next run. So, if you want to change the data-options in a config file, you need to delete the corresponding processed files first. This is because the code will search `.pt` files given the `setting_name`; if it finds any `.pt` files under `$ProjectDir/data/processed-[setting_name]-[cut_id]`, it will assume that the processed files for the specified setting are already there and will not re-process data with the new options. If you've only changed options under 'model' or 'optimizer', you do you need to delete the processed files.

# Workflow of the code

1. Class `Tau3MuDataset` in `$ProjectDir/src/utils/dataset_splitz` is used to build datasets that can be used to train pytorch models. The code will first call this class to process dataframes, including graph building, node/edge feature generations, dataset splits, etc. After this process, the fully processed data shall be saved on the disk.

2. Then the model will be trained by the class `Tau3MuGNNs` in `train_gnn_parallel.py`, and during the training some metrics will show on the progress bar. Currently, the model must be run with 2 gpus available. This will be fixed in the near-future.

3. The trained model will be saved into `$ProjectDir/data/logs/[time_step-setting_name-cut_id]/model.pt`, where `[time_step-setting_name-cut_id]` is the log id for this model and will be needed to load the model later.


# Training Logs
Standard output provides basic training logs, while more detailed logs and interpretation visualizations can be found on tensorboard:
```
python -m tensorboard.main dev upload --logdir=$LogDir
```
