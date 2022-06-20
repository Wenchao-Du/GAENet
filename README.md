# Depth Completion using geometry-aware embedding
This repo is the Pytorch implementation for our paper accepted by ICRA22 on ["Depth Completion using Geometry-aware Embedding"](https://arxiv.org/abs/2203.10912), developed by Wenchao Du, Hu Chen, Hongyu Yang and Yi Zhang at Sichuan University.

Our method is trained with kitti and NYUv2 dataset, and achieves the signficant performance gains without specifical designing.



# Contents
1. Dependency
2. Data
3. Pretrained models
4. Training and testing
5. Citation
   
# Dependency
This code was tested with Python3 and Pytorch >=1.0 on Ubuntu 16.04 and above

# Data
1. For outdoor environment, downloading [KITTI Depth Completion Data](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) from their website.
2. For indoor environment, downloading [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset.

Format your data following the data_json and configs.

# Pretrained models
The original pretrained models are put into the pretrained_models folder, you could use it for validation.

# Citation
If you use this code or method in your work, please cite the following:
```
@article{Du2022DepthCU,
	title={Depth Completion using Geometry-Aware Embedding},
	author={Wenchao Du, Hu Chen, Hongyu Yang and Yi zhang},
	booktitle={ICRA},
	year={2022}
}
```


