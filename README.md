# SACNet
Code repository for our paper entilted "Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark" accepted at TMM 2024.

arXiv version: https://arxiv.org/abs/2406.00917.

## :tada: **News** :tada:  (December, 2024)

We are excited to announce that our new work **"Alignment-Free RGB-T Salient Object Detection: A Large-scale Dataset and Progressive Correlation Network"** has been accepted to **AAAI 2025**! :link: [GitHub Repository: PCNet](https://github.com/Angknpng/PCNet)

This is also part of our ongoing **Alignment-Free RGB-T Salient Object Detection** series. Stay tuned for updates regarding code and resources related to this new work. 🚀

## Citing our work

If you think our work is helpful, please cite

```
@article{Wang2024alignment,
  title={Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark},
  author={Wang, Kunpeng and Lin, Danying and Li, Chenglong and Tu, Zhengzheng and Luo, Bin},
  journal={IEEE Transactions on Multimedia},
  year={2024}
}
```
## The Proposed Unaligned RGBT Salient Object Detection Dataset

### UVT2000

We construct a novel benchmark dataset, containing 2000 unaligned visible-thermal image pairs directly captured from various real-word scenes, to facilitate research on alignment-free RGBT SOD.

[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/dataset_sample.png)](https://github.com/Angknpng/SACNet/blob/main/figures/dataset_sample.png)

The proposed dataset link can be found here. [[baidu pan](https://pan.baidu.com/s/1tLYnRAMXANvEB4qUM1jZgw?pwd=nitk) fetch code: nitk] or [[google drive](https://drive.google.com/drive/folders/1Rm-zZRIAJmBhyS71WGKVL4IsrziR70bo?usp=drive_link)]

### Dataset Statistics and Comparisons

We analyze the proposed UVT2000 datset from several statistical aspects and also conduct a comparison between UVT2000 and other existing multi-modal SOD datasets.

[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/dataset_compare.png)](https://github.com/Angknpng/SACNet/blob/main/figures/dataset_compare.png)

## Overview
### Framework
[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/framework.png)](https://github.com/Angknpng/SACNet/blob/main/figures/framework.png)
### RGB-T SOD Performance
[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/performance_RGBT.png)](https://github.com/Angknpng/SACNet/blob/main/figures/performance_RGBT.png)
### RGB-D SOD Performance
[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/performance_RGBD.png)](https://github.com/Angknpng/SACNet/blob/main/figures/performance_RGBD.png)
### RGB SOD Performance
[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/performance_RGB.png)](https://github.com/Angknpng/SACNet/blob/main/figures/performance_RGB.png)

## Predictions

RGB-T saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1uuDHAh9TTQ4N9cJIl-p9OQ?pwd=kgej) fetch code: kgej] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

RGB-D saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/15V1RYOUDFnx2w-GXDjzv8Q?pwd=43bk) fetch code: 43bk] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

RGB saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1MMmOOrTG7D_o3iDCihpdCQ?pwd=8ug9) fetch code: 8ug9] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

ResNet50-based saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/14HFp58DmvjQHrLNSqz7uwA?pwd=i8xg) fetch code: i8xg]

ResNet50-based checkpoints can be found here. [[baidu pan](https://pan.baidu.com/s/1Gkkp3R7gq4-Slxqfr0BeVw?pwd=istd) fetch code: istd]

## Pretrained Models
The pretrained parameters of our models can be found here. [[baidu pan](https://pan.baidu.com/s/177h2BnBwJ2C81qwVeC4Z0g?pwd=f2x5) fetch code: f2x5] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

## Usage

### Requirement

0. Download the datasets for training and testing from here. [[baidu pan](https://pan.baidu.com/s/1RE48go1wzGWymMblawG2wQ?pwd=vvgq) fetch code: vvgq]
1. Download the pretrained parameters of the backbone from here. [[baidu pan](https://pan.baidu.com/s/1sBuu7Qw9n8aWRydQsDieBA?pwd=3ifw) fetch code: 3ifw]
2. Create directories for the experiment and parameter files.
3. Please use `conda` to install `torch` (1.12.0) and `torchvision` (0.13.0).
4. Install other packages: `pip install -r requirements.txt`.
5. Set your path of all datasets in `./Code/utils/options.py`.

### Train

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2212 train_parallel.py
```

### Test

```
python test_produce_maps.py
```

## Acknowledgement

The implement of this project is based on the following link.

- [SOD Literature Tracking](https://github.com/jiwei0921/SOD-CNNs-based-code-summary-)
- [PR Curve](https://github.com/lartpang/PySODEvalToolkit)
- [Computational complexity test](https://github.com/yuhuan-wu/MobileSal)

## Contact

If you have any questions, please contact us (kp.wang@foxmail.com).
