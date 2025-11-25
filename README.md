# STOF AE

https://zenodo.org/records/17705801

This folder contains the system prototype of STOF (pap161) at PPoPP '26, titled "Accelerating Sparse Transformer Inference on GPU", including Figure 9_10, Figure 11, Figure 12, Figure 13 and Table 4.

## Abstract


The repository is organized as below:

- `data/`: original data log in `data/MHA_performance` for Figure 9_10, `data/End2End_performance` for Figure 11, `data/Ablation_Study` for Figure 12, `data/Overhead_Analysis` for Figure 13. `data/Tuning_Cost` for Table 4.

- `plot/`: quick plotting reproduction code to get the figures in the paper, including `fig3/`, `fig4/`, `fig9_10/`, `fig11/`, `fig12/`, and `fig13/`.

- `script/`: `.sh` executable script to install the custom operator in STOF and execute it in full to reproduce the experimental results in the paper. Including `env_install`, `fig9_10.sh`, `fig11.sh`, `fig12.sh`, and `fig13.sh`.

- `src/`: The core source code implemented in STOF, especially the unified MHA kernels, is in `src/ops/src/***.cu` bound by `src/setup.py`. The baselines that can be run directly include PyTorch Native, PyTorch Compiled, SPLAT, ByteTransformer, FlashAttention2, and FlexAttention. MCFuser and Bolt need to be executed separately due to the complex compilation environment, which will be introduced later.

## Getting Started

### Log into the provided server

A ssh private key is provided for AE reviewers, named `id_rsa_ppopp26_ae`, to access the provided server.

```shell
# log into the provided server (A100) 
ssh -i ./id_rsa_ppopp26_ae -o IdentitiesOnly=yes -o ProxyCommand="ssh -i ./id_rsa_ppopp26_ae -o IdentitiesOnly=yes -p 6000 -W %h:%p 19762@8.218.213.105" -p 45005 sunqingxiao@10.254.46.24

# enter the container
docker exec -it ppopp-stof /bin/bash

# enter the workspace
cd /PPoPP26-pap161-AE
```

The logged server configured with NVIDIA A100 GPU. By `ssh device4090`, reviewers can log into another server configured with NVIDIA RTX 4090 GPU.

### Use `tmux` for Long-Running Scripts
 It is strongly recommended to run all scripts inside through [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) to prevent interruptions.
```shell
# create a session named 'ppopp26_pap161_ae'
tmux new -s ppopp26_pap161_ae

# now we are in the session
RUN/PROVIDED/SCRIPTS

# detach session: Type Ctrl + b, then D
Ctrl-b D

# list all sessions
tmux ls

# reattach to a session named 'ppopp26_pap161_ae'
tmux a -t ppopp26_pap161_ae
```

### Quick Reproduction: Plot from Backup Logs (~2 minutes)

To quickly reproduce the figures in the proposed paper, the backup logs are provided for plotting.
```shell
# enter script directory
cd script

# batch plot script executor
bash run_all_plots.sh
```

### In-depth Reproduction: Plot from Real Run (~2 hours)

To actually execute the program and plot the results on the server, follow the steps below. Due to the relatively complex dependencies and configuration files required by MCFuser and Bolt, we have not included these experimental components in this repository. To obtain the complete data as presented in the original paper, please follow the instructions under "Comparisons that need to be run separately in the Artifact" for comprehensive execution.

```shell
# enter script directory
cd script

# for Figure 9_10 (~10 minutes)
bash fig9_10.sh

# for Figure 11 without Bolt (~30 minutes)
bash fig11.sh

# for Figure 12 (~30 minutes)
bash fig12.sh

# for Figure 13 (~10 minutes)
bash fig13.sh

# for STOF in Table 4 (~1 hour) 
bash table4_STOF.sh
```


### Installation

It is recommended to use the provided server to avoid network issues and complex dependencies. For deploying on your own server, you can pull an image `nvcr.io/nvidia/pytorch:24.09-py3` to obtain the basic environment.

```shell
# pull docker image and enter container
docker pull nvcr.io/nvidia/pytorch:24.09-py3
docker run --gpus all --name ae-env --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nvcr.io/nvidia/pytorch:24.09-py3 /bin/bash

# if build from a more basic environment without using the recommended image above, please install PyTorch 2.7.0
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# clone the repository and enter the directory
git clone https://github.com/HeyDavid633/PPoPP26-pap161-AE.git
cd PPoPP26-pap161-AE

# enter script directory
cd script
# install operators and check the environment
# according to running device input sm_{CUDAARCH}, e.g.,  A100:sm_80 4090:sm_89,
# so that for A100: bash env_install.sh 80, and for 4090: bash env_install.sh 89
bash env_install.sh 80
```

### Comparisons that need to be run separately in the Artifact

For the comparison of baselines MCFuser and Bolt, we have uploaded the relevant necessary configuration files to  [Google Drive](https://drive.google.com/file/d/17N-PfI0klMa1jHE-1YcpV5oNzjfcFxE4/view?usp=sharing). After downloading them and place the compressed package `ae-mcfuser-test.tar.gz` in `/src`, you need to execute the relevant installation script `script/MCFuser_install.sh`. The detailed steps are as follow:

```shell
cd PPoPP26-pap161-AE/src

# download ae-mcfuser-test.tar.gz from Google Drive
# uncompressed this file
tar -xzvf ae-mcfuser-test.tar.gz

# rename this directory
mv ae-mcfuser-test3 ./MCFuser/mcfuser
cd ../script

# install MCFuser and Bolt
bash MCFuser_install.sh

# for MCFuser and Bolt in Table 4 (~1 hour)
bash table4.sh
```
