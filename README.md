# 1.简介
基于detectron2的PanopticFPN模型仓库

# 2.环境配置
## Requirements
- PyTorch ≥ 1.4 and torchvision that matches the PyTorch, 参考[pytorch.org](https://pytorch.org/)
- OpenCV is optional and needed by demo and visualization, 参考`environment.yml`

## Build Detectron2 from Source
gcc & g++ ≥ 5 are required. ninja is recommended for faster build. After having them, run:

```Python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# Or if you are on macOS
CC=clang CXX=clang++ python -m pip install ......
```
ps: To rebuild detectron2 that's built from a local clone, use rm -rf build/ **/*.so to clean the old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

# 3.训练流程
## 制作COCO格式数据集
参考脚本：
- `custom_2_coco_format/scripts/custom_2_coco_instance_format.py`: 生成instance任务的COCO格式数据
- `custom_2_coco_format/scripts/custom_2_coco_panoptic_format.py`: 生成panoptic任务的COCO格式数据
- `datasets/prepare_panoptic_fpn.py`: 生成panoptic所需的sementaic segmentation mask

## registe custom dataset
参考脚本：
- `detectron2/data/datasets/builtin.py`:

## 配置参数文件
参考配置文件：
- `configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_card.yaml`: 卡片任务的配置文件
- `configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_text.yaml`: 混合文本检测的配置文件

## train/evaluate/visualize/interface
train:
```Python
python3 train_net.py --config-file ../configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_text.yaml --num-gpus 1
```

evaluate:
```Python
python3 train_net.py --config-file ../configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_text.yaml --eval-only MODEL.WEIGHTS /media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/train_results/panoptic_seg/detectron2/text_20200724/model_final.pth OUTPUT_DIR /media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/train_results/panoptic_seg/detectron2/text_20200724/eval
```

visualize results:
- `tools/inference.py`

interface:
- `interface/en_cn_text/panoptic_fpn_mix_text.py`

# 4.相关实验结果
- 参考文档：[全景分割模型试验对比](https://gostudyai.feishu.cn/docs/doccnHLX4JpFGYaHVedF10Ayl9p#)

# 5.Reference
- [detectron2](https://github.com/facebookresearch/detectron2)