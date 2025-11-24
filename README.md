# STOF AE

==(最后上传后放 zenodo 链接)==

This folder contains the system prototype of STOF (pap161) at PPoPP '26, titled "Accelerating Sparse Transformer Inference on GPU", including Figure 9_10, Figure 11, Figure 12, Figure 13 and Table 4.

## Abstract


The repository is organized as below:

- `data/`: orignal data log in `data/MHA_performance` for Figure 9_10, `data/End2End_performance` for Figure 11, `data/Ablation_Studdy` for Figure 12, `data/Overhead_Analysis` for Figure 13. `data/Tuning_Cost` for Table 4.

- `plot/`: quick poltting reproduction code to get the images in the paper, including `fig3/`, `fig4/`, `fig9_10/`, `fig11/`, `fig12/`, and `fig13/`.

- `script/`: `.sh` executable script to install the custom operator in STOF and execute it in full to reproduce the experimental results in the paper. Including `env_install`, `fig9_10.sh`, `fig11.sh`, `fig12.sh`, and `fig13.sh`.

- `src/`: The core source code implemented in STOF, especially the unified MHA kernels, is in `src/ops/src/***.cu` bound by `src/setup.py`. The baselines that can be run directly include PyTorch Native, PyTorch Compiled, SPLAT, ByteTransformer, FlashAttention2, and FlexAttention. MCFuser and Bolt need to be executed separately due to the complex compilation environment, wihich will be introduced later.

## Getting Started

### Log into the provided cluster

==待补充: 
1.在北航机器4090/A100 上新开账户为ppopp26_pap161_ae 
2.机器应该有公开网段, 当前A100网段10.254.46.24不公开; 需北航 VPNgit 
3.并在北航 A100 机器上配置快速登录 4090 的方式，从而只用给一个 ssh key==

A ssh private key is provided for AE reviewers, named `id_rsa_ppopp26_pap161_ae`, to access the provided cluster.

```shell
ssh -p XXXX -i ./id_rsa_ppopp26_pap161_ae ppopp26_pap161_ae@XX.XX.XX.XX   # A100
```

The logged cluster is named `ppopp26_pap161_ae_reviewer`,  which is configured with NVIDIA A100 GPU. By `ssh PPoPP26_pap161_ae_device4090`, reviewers can log into another cluster named `PPoPP26_pap161_ae_device4090`,  which is configured with NVIDIA RTX 4090 GPU.

### Use `tmux` for Long-Running Scripts
It is strongly recommend that running all scripts inside through [tmux](https://github.com/tmux/tmux/wiki/Getting-Started) to prevent interruptions.
```shell
# Create a session named 'ppopp26_pap161_ae'
tmux new -s ppopp26_pap161_ae
# Now we are in the sesion
RUN/PROVIDED/SCRIPTS
# Detach session: Type Ctrl + b, then D
Ctrl-b D
# List all sessions
tmux ls
# Reattach to a session named 'ppopp26_pap161_ae'
tmux a -t ppopp26_pap161_ae
```

### Quick Reproduction: Plot from Backup Logs (~1 minutes)

To quickly reproduce the figures in the proposed paper, the backup logs are provided for plotting.
```shell
# enter script directory
cd script

#  Batch plot script executor
bash run_all_plots.sh
```

### In-depth Reproduction: Plot from Real Run (~3 hours)

To execute the program to obtain hot data and plot the results on site。

```shell
# enter script directory
cd script

# for Figure 9_10
bash fig9_10.sh

# for Figure 11
bash fig11.sh

# for Figure 12
bash fig12.sh

# for Figure 13
bash fig13.sh

# for STOF in Table 4
bash table4_STOF.sh
```


### Installation

It is recommend that using the provided machine to avoid some network issues and complex dependencies regarding MCFuser and Bolt. If reviewers want to deploy on their own machine, it is recommended to use an image `nvcr.io/nvidia/pytorch:24.09-py3` to directly obtain the container with the basic environment.

```shell
# pull docker images and enter conatiner
docker pull nvcr.io/nvidia/pytorch:24.09-py3
docker run --gpus all --name ae-env --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nvcr.io/nvidia/pytorch:24.09-py3 /bin/bash

# clone the repository and encter the directory
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

For the comparison of Blselines MCFuser and Bolt, a lot of compilation and installation processes related to tvm and CUTLASS are involved. In order to reproduce this part of the experiment smoothly, we have uploaded the relevant necessary configuration files to [Google Drive](https://drive.google.com/file/d/17N-PfI0klMa1jHE-1YcpV5oNzjfcFxE4/view?usp=sharing). After downloading them and place the compressed package `ae-mcfuser-test.tar.gz` in `/src`, you need to execute the relevant installation script `script/MCFuser_install.sh`. The exact steps are as follows:

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

# for MCFuser and Bolt in Table 4
bash table4.sh
```
