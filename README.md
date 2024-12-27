# SACNet
Code repository for our paper entilted "Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark" accepted at TMM 2024.

arXiv version: https://arxiv.org/abs/2406.00917.
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

The proposed dataset link can be found here. [[baidu pan](https://pan.baidu.com/s/1S2IFZjmWNf2EQtMVk5q2yg?pwd=irwv) fetch code: irwv] or [[google drive](https://drive.google.com/drive/folders/1Rm-zZRIAJmBhyS71WGKVL4IsrziR70bo?usp=drive_link)]

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

RGB-T saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1CSa8HNzlW9Lq4ToXlmuZnQ?pwd=xyu7) fetch code: xyu7] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

RGB-D saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1dm-PbN_WMmVCRovJr3yROQ?pwd=jrjl) fetch code: jrjl] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

RGB saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1cvDj92KWVP8XL8DsPeA9ig?pwd=kj6o) fetch code: kj6o] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

ResNet50-based saliency maps can be found here. [[baidu pan](https://pan.baidu.com/s/1vJwjQeEHS18MWNscDfwrAw?pwd=xm4c) fetch code: xm4c]

ResNet50-based checkpoints can be found here. [[baidu pan](https://pan.baidu.com/s/1o1DrfCL4T17PbKtU4b3mAw?pwd=svvk) fetch code: svvk]

## Pretrained Models
The pretrained parameters of our models can be found here. [[baidu pan](https://pan.baidu.com/s/1LrGbcrkqkdMSgjx0C5S1hg?pwd=ihri) fetch code: ihri] or [[google drive](https://drive.google.com/drive/folders/1fBS4GMBS5qja8pzg7ZCQd1AxmEC8rMun?usp=drive_link
)]

## Usage

### Requirement

0. Download the datasets for training and testing from here. [[baidu pan](https://pan.baidu.com/s/1V6bPH87yZZ2fRbfa62vXeg?pwd=075x) fetch code: 075x]
1. Download the pretrained parameters of the backbone from here. [[baidu pan](https://pan.baidu.com/s/14xGtKVSs53zRNZVKK-x4HA?pwd=mad3) fetch code: mad3]
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
