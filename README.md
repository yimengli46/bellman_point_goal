# lsp-habitat
This repository contains code for an agent navigating to a point goal via LSP on Habitat environments.
<img src='Figs/example_traj.jpg'>
### Installation
```
git clone --branch yimeng https://github.com/RAIL-group/lsp-habitat.git
cd  lsp-habitat
mkdir output
```

### Dependencies
We use `python==3.7.4`.  
We recommend using a conda environment.  
```
conda create --name lsp_habitat python=3.7.4
source activate lsp_habitat
```
You can install Habitat-Lab and Habitat-Sim following guidance from [here](https://github.com/facebookresearch/habitat-lab "here").  
We recommend to install Habitat-Lab and Habitat-Sim from the source code.  
We use `habitat==0.2.1` and `habitat_sim==0.2.1`.  
Use the following commands to set it up:  
```
# install habitat-lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/stable
pip install -e .

# install habitat-sim
git clone --recurse --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
sudo apt-get update || true
# These are fairly ubiquitous packages and your system likely has them already,
# but if not, let's get the essentials for EGL support:
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
git checkout tags/stable
python setup.py install --with-cuda
```
You also need to install the dependencies:  
```
habitat==0.2.1
habitat-sim==0.2.1
torch==1.8.0
torchvision==0.9.0
matplotlib==3.3.4
networkx==2.6.3
scikit-fmm==2022.3.26
scikit-image
sknw
tensorboardX
```
To install lsp-accel, change `line 30` of `RAIL-core-main/modules/lsp_accel/CMakeLists.txt` into where *Eigen* is located.
```
pip install RAIL-core-main/modules/lsp_accel
```

### Dataset Setup
Download *scene* dataset of **Matterport3D(MP3D)** from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md "here").      
Upzip the scene data and put it under `habitat-lab/data/scene_datasets/mp3d`.  
You are also suggested to download *task* dataset of **Point goal Navigation on MP3D** from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md "here")  
Unzip the episode data and put it under `habitat-lab/data/datasets/pointnav/mp3d`.  
Create softlinks to the data.  
```
cd  lsp-habitat
ln -s habitat-lab/data data
```
The code requires the datasets in data folder in the following format:
```
habitat-lab/data
                /datasets/pointnav/mp3d/v1
                                        /train
                                        /val
                                        /test
                scene_datasets/mp3d
                                    /1LXtFkjw3qL
                                    /1pXnuDYAj8r
                                    /....
```

### How to Run?
The code can do  
(a) Point Goal Navigation on MP3D test episodes.   
All the parameters are controlled by the configuration file `core/config.py`.   
Please create a new configuration file when you initialize a new task and saved in folder `configs`.
##### Exploring the environment
To run the large-scale evaluation, you need to download pre-generated `scene_maps`, `scene_floor_heights` and `large_scale_semantic_maps` from [here](https://drive.google.com/file/d/1uqCL6N2kpOPjvumw-lBQw55bv288qkDx/view?usp=share_link "here").  
Download it, unzip it and put the folders under `lsp-habitat/output`.  
Then you can start the evaluation.  
For example, if you want to evaluate the optimistic planner on your desktop, use the following command.  
```
python main_eval.py --config='exp_90degree_Optimistic_PCDHEIGHT_MAP_1STEP_500STEPS.yaml'
```
If you have a server with multiple GPUs, use another configuration file.
```
python main_eval_multiprocess.py --config='large_exp_90degree_Optimistic_PCDHEIGHT_MAP_1STEP_500STEPS.yaml'
```


