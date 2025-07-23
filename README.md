**IndoLayout: Leveraging Attention for Extended Indoor Layout Estimation from an RGB Image**  
Shantanu Singh, Jaidev Shriram, Shaantanu Kulkarni, Brojeshwar Bhowmick, K. Madhava Krishna  

[IROS 2022](https://ieeexplore.ieee.org/document/9982106) • [Project Website](https://indolayout.github.io) • [Paper PDF](https://drive.google.com/file/d/1E918al3PJ7f1jvyf4vB_PDy9r0SSq1GY/view?usp=sharing)

IndoLayout is a lightweight and real-time deep learning model that estimates **amodal indoor layouts** from a single monocular RGB image. Unlike traditional methods that only detect visible occupancy, IndoLayout leverages **self-attention** and **adversarial learning** to predict the **hallucinated occupancy** of occluded spaces, enabling better mapping and planning for indoor robots.

<p align="center">
  <img src="https://indolayout.github.io/resources/teaser.png" width="700"/>
</p>

This repository contains the official implementation of IndoLayout, a method for extended indoor layout estimation from a single RGB image. IndoLayout leverages attention mechanisms to improve the accuracy and robustness of layout predictions in complex indoor scenes.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/indoor-layout-estimation.git
cd indoor-layout-estimation
pip install -r requirements.txt
```

## 🧠 Model Overview

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

## Training
To train IndoLayout on your dataset, use one of the provided training scripts. For example:

```sh
bash train_posenet.sh
```

Or directly with Python:

```sh
python train_posenet.py--config configs/your_config.yaml
```

Please note that this runs both training on the train set and evaluation on the validation set.

You can find additional configs to train different baselines in the repository.

## Evaluation
For evaluation, use the ``eval_*.yaml`` files in the configs directory. Alternatively, simply set the config ```PIPELINE.train=[]```  for the same result.

You can then use the ```train_posenet.py``` script to run only evaluation.


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

## 🧪 Citation

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
This project is licensed under the MIT License. See the LICENSE file for details.


## Acknowledgements
Developed at the [Robotics Research Center](https://robotics.iiit.ac.in/), IIIT-Hyderabad, in collaboration with TCS Research. Portions of the code may be adapted from other open-source projects. Please see individual files for attribution.