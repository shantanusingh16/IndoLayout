<div align="center">
<h1>Indolayout: Amodal indoor layout estimation</h1>

<a href="https://ieeexplore.ieee.org/document/9982106" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Conference-green" alt="Paper PDF">
</a>
<a href="https://drive.google.com/file/d/1E918al3PJ7f1jvyf4vB_PDy9r0SSq1GY/view?usp=sharing"><img src="https://img.shields.io/badge/PDF-orange" alt="PDF"></a>
<a href="https://indolayout.github.io"><img src="https://img.shields.io/badge/Project%20Page-blue" alt="Project Page"></a>


**[RRC, IIIT Hyderabad](https://robotics.iiit.ac.in/)**; **[TCS](https://www.tcs.com/what-we-do/research)**

</div>

## Overview
IndoLayout is a lightweight and real-time deep learning model that estimates **amodal indoor layouts** from a single monocular RGB image. Unlike traditional methods that only detect visible occupancy, IndoLayout leverages **self-attention** and **adversarial learning** to predict the **hallucinated occupancy** of occluded spaces, enabling better mapping and planning for indoor robots.

<p align="center">
  <img src="https://indolayout.github.io/resources/teaser.png" width="700"/>
</p>

## Description
This repository contains the official implementation of IndoLayout, a method for extended indoor layout estimation from a single RGB image. IndoLayout leverages attention mechanisms to improve the accuracy and robustness of layout predictions in complex indoor scenes.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Baselines](#baselines-ans-and-occant)
- [Indolayout](#indolayout)
- [Supplementary](#supplementary-material)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/indoor-layout-estimation.git
cd indoor-layout-estimation
pip install -r requirements.txt
```

## Model Overview

IndoLayout uses a combination of:
- **ResNet-18 encoder** (pretrained on ImageNet)
- **Transformer-style self-attention module**
- **Convolutional decoder**
- **Patch-based discriminator (GAN)**

The model outputs a **3-channel top-down occupancy map** with probabilities for:
- Free/Navigable space
- Occupied space
- Unknown/Unexplored area

<p align="center">
  <img src="https://indolayout.github.io/resources/method_diagram.png" width="720"/>
</p>

## Dataset Preparation
We train and evaluate on the following photorealistic indoor datasets using the [Habitat simulator](https://aihabitat.org):
- Gibson 4+ (train/val)
- Gibson Tiny (train/val)
- Matterport3D (val only)
- HM3D (val only)

The ```generate_habitat_data.py``` script in the repository is for getting a quick overview.

For generating datasets, please use the actual Habitat repositories here: [Github](!https://github.com/indolayout)

## Baselines (ANS and OccAnt)
Please refer to the bash_scripts folder to see examples on how to train or evaluate the Indolayout model on different datasets.

### Environment variables
To run the code, you need to setup environment variables, which can be done by sourcing the 'vars' file in bash_scripts folders. 

<b>Pattern:</b>

```bash
source bash_scripts/gibson4_exp.vars {Dataset_dir} {Bev_dir} {Log_Dir} {Train_split_path} {Val_split_path}
```


<b>E.g.:</b>

```bash
source bash_scripts/gibson4_exp.vars /home/$USER/gibson4_dataset /home$USER/gibson4_dataset/bev_dir /home/$USER/indolayout_logs /home/$USER/indoor-layout-estimation-main/splits/gibson4/filtered_front_train_files.txt /home/$USER/indoor-layout-estimation-main/splits/gibson4/filtered_front_val_files.txt
```

### Training ANS RGB or Occant RGB/RGB-D
The main script for training the baseline models (Occupancy Anticipation RGB and RGB-D) is:

```bash
train_posenet_bev.py train_occant_gibson4.yaml --script_mode train
```

Please note that this runs both training on the train set and evaluation on the validation set.

You can find additional configs to train different baselines in the repository.

### Evaluate ANS RGB or Occant RGB/RGB-D
To evaluate the same model, simply change the script_mode to 'val' as follows:

```bash
train_posenet_bev.py train_occant_gibson4.yaml --script_mode val
```

## Indolayout

### Environment variables
Similar to the baselines, to run the code, you need to setup environment variables, which can be done by sourcing the 'vars' file in bash_scripts folders. 

<b>Pattern:</b>

```bash
source cross-view/gibson4_exp.vars {Dataset_dir} {Bev_dir} {Log_Dir} {Train_split_path} {Val_split_path}
```


<b>E.g.:</b>

```bash
source cross-view/gibson4_exp.vars /home/$USER/gibson4_dataset /home/$USER/gibson4_dataset/dilated_partialmaps /home/$USER/indolayout_logs /home/$USER/indoor-layout-estimation-main/splits/gibson4/filtered_front_train_files.txt /home/$USER/indoor-layout-estimation-main/splits/gibson4/filtered_front_val_files.txt
```

### Training Indolayout model
The main script for training the indolayout model is:

```bash
python3 train_disc.py --model_name attention_transformer_discr --data_path /home/$USER/gibson4_dataset --split gibson4 --width 512 --height 512 --num_class 3 --type static --static_weight 1 --occ_map_size 128 --log_frequency 1 --log_root /home/$USER/basic_discr --save_path /home/$USER/basic_discr --semantics_dir None --chandrakar_input_dir None --floor_path None --batch_size 8 --num_epochs 100 --lr_steps 50 --lr 1e-4 --lr_transform 1e-3 --load_weights_folder None --bev_dir /home/$USER/gibson4_dataset/dilated_partialmaps --train_workers 15 --val_workers 8
```

### Evaluate Indolayout
To evaluate the same model, simply change the script to 'eval.py' as follows:

```bash
eval.py --model_name attention_transformer_discr --data_path /home/$USER/gibson4_dataset --split gibson4 --width 512 --height 512 --num_class 3 --type static --static_weight 1 --occ_map_size 128 --log_frequency 1 --log_root /home/$USER/basic_discr --load_weights_folder /home/$USER/basic_discr/epoch_100 --semantics_dir None --chandrakar_input_dir None --floor_path None --batch_size 8 --num_epochs 1 --bev_dir /home/$USER/gibson4_dataset/dilated_partialmaps --train_workers 0 --val_workers 8
```

## Supplementary Material

Refer to the notebooks folder to understand experiments and their implementations individually, in particular for:
1. data_visualization
2. generate_visible_occupancy
3. photometric_reconstruction
4. evaluate_bev

## Results
### Gibson 4+ (Validation)

| Method           | mIoU (%) | F1 (%) | SSIM  | Boundary IoU (%) |
|------------------|----------|--------|-------|------------------|
| ANS (RGB) [5]    | 32.85    | 47.03  | 49.23 | 14.65            |
| OccAnt (RGB) [27]| 57.09    | 71.07  | 66.19 | 36.42            |
| **IndoLayout**   | **63.45**| **73.49** | **69.37** | **39.06**  |

### PointNav Task (Gibson 4+)

| Difficulty | Method           | Success Rate ↑ | SPL ↑  | Time ↓   |
|------------|------------------|----------------|--------|----------|
| Easy       | IndoLayout       | **0.913**      | 0.731  | 127.10s  |
| Medium     | IndoLayout       | **0.763**      | 0.566  | 233.80s  |
| Hard       | IndoLayout       | **0.431**      | 0.337  | 383.33s  |

IndoLayout improves navigation success across difficulty levels without any depth input.

## Citation

If you use IndoLayout in your research, please cite:

```bibtex
@INPROCEEDINGS{9982106,
  author={Singh, Shantanu and Shriram, Jaidev and Kulkarni, Shaantanu and Bhowmick, Brojeshwar and Krishna, K. Madhava},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={IndoLayout: Leveraging Attention for Extended Indoor Layout Estimation from an RGB Image}, 
  year={2022},
  volume={},
  number={},
  pages={13128-13135},
  keywords={Measurement;Layout;Estimation;Real-time systems;Adversarial machine learning;Indoor environment;Task analysis},
  doi={10.1109/IROS47612.2022.9982106}}

```

## License
This project is licensed under the MIT License. See the [LICENSE]() file for details.


## Acknowledgements
Developed at the [Robotics Research Center](https://robotics.iiit.ac.in/), IIIT-Hyderabad, in collaboration with [TCS Research](https://www.tcs.com/what-we-do/research). 

Thanks to these great repositories: [Neural-SLAM](https://github.com/devendrachaplot/Neural-SLAM), [Occupancy Anticipation](https://github.com/facebookresearch/OccupancyAnticipation), [Cross-View](https://github.com/JonDoe-297/cross-view) and many other inspiring works in the community.