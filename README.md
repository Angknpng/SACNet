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

The proposed dataset link can be found here. [[baidu pan](https://pan.baidu.com/s/1S2IFZjmWNf2EQtMVk5q2yg?pwd=irwv) fetch code: irwv]

### Dataset Statistics and Comparisons

We analyze the proposed UVT2000 datset from several statistical aspects and also conduct a comparison between UVT2000 and other existing multi-modal SOD datasets.

[![avatar](https://github.com/Angknpng/SACNet/raw/main/figures/dataset_compare.png)](https://github.com/Angknpng/SACNet/blob/main/figures/dataset_compare.png)

## Predictions

RGB-T saliency maps can be found here. 
RGB-D saliency maps can be found here. 
RGB saliency maps can be found here. 

## Pretrained Models
Pretrained parameters can be found here.

## Usage

### Requirement

0. Download the datasets for training and testing from here.
1. Create directories for the experiment and parameter files.
2. Please use `conda` to install `torch` (1.12.0) and `torchvision` (0.13.0).
3. Install other packages: `pip install -r requirements.txt`.
4. Set your path of all datasets in `./Code/utils/options.py`.

### Train

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2212 train_parallel.py
```

### Test

```
python test_produce_maps.py
```

## Contact

If you have any questions, please contact us (kp.wang@foxmail.com).
