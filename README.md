# Active Learning for Deep Object Detection
This repo contains the code that I used to write my undergraduate thesis on active learning for deep object detection. The code in the repo was adapted from [sgrvinod's tutorial on object detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#base-convolutions--part-1). If you find any errors or room for improvements please do not hesitate to leave a pull request. 

## Setup Instructions

### Clone and install requirements

```console
git clone https://github.com/CtfChan/pytorch_ssd_active_learning.git
cd pytorch_ssd_active_learning
conda env create -f environment.yml
conda activate tf_gpu
```

### Prepare dataset
```console
cd pytorch_ssd_active_learning/voc
bash download_voc.bash #this will take a while
python3 create_data_lists.py #this will save the paths of the images to a json file
```

### Run Experiments
```console
python3 train_baseline.py # this will take a while
bash generate_plots.bash # this takes even longer
```

### Data visualizations
The experiments will be saved in to the ```results/``` directory. You can run any of the .ipynb files to recreate the plots that were used in my thesis. 
