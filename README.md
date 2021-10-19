# Weighted Boxes Fusion with Detectron2

This project is an application of Weighted Boxes Fusion (WBF) to ensemble multiple detection modules. Specifically, we ensemble 3 object detection models available on detectron2's model zoo. With each module detecting objects from an image, WBF mixes the detection results to acquire a more robust and accurate pipeline.


## Prerequisites

0. Conda create env
```sh
# create new env named "sv"
conda create --name sv
```

1. [Detectron2 (:=pytorch framework)](https://github.com/facebookresearch/detectron2)
```sh
# Install as instructed in the link below
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html

# OR if you have a GPU compatible with CUDA 11.1, follow the steps below.
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```
btw, [Detectron Colab Tutorial](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=FsePPpwZSmqt) is a great way to get familiar with Detectron 2.

2. [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
```sh
pip3 install ensemble-boxes # or just pip
```

3. install OpenCV and tqdm
```sh
conda install opencv -c conda-forge
conda install tqdm
```

3. Git clone this repository
```sh
git clone git@github.com:grandoba/detectron2_ensemble.git
```

4. Download the COCO datasets [ref](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)
```sh
cd detectron2_ensemble
mkdir datasets && cd datasets
mkdir coco && cd coco

# download validation dataset
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
rm val2017.zip

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip

```

## Reproducing the results
```sh
# first run mode=1 to get the results from each model and save to directory "outputs1", "outputs2", and "outputs3"
python3 main.py --mode 1

# Run WBF ensemble: AP@[0.50:0.95] = 43.30
python3 main.py --mode 5
```

## Results

| Model   |   AP       |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------: |:------:    |:------:|:------:|:------:|:------:|:------:|
| model 1 | 36.816     | 55.767 | 40.476 | 18.179 | 41.358 | 50.503 |
| model 2 | 40.397     | 60.251 | 43.185 | 24.022 | 44.345 | 52.185 |
| model 3 | 39.614     | 56.974 | 43.858 | 22.577 | 42.940 | 52.098 |
| Ensemble| **43.299** | 63.993 | 47.095 | 27.171 | 47.716 | 54.641 |

* model 1: faster_rcnn_R_50_DC5_3x

* model 2: retinanet_R_101_FPN_3x

* model 3: faster_rcnn_X_101_32x8d_FPN_3x

* Ensemble: Weighed Boxes Fusion

More detailed results are in the `results/` folder.

Below is the original image.
<p align="center">
   <img src="imgs/img_original.png" alt="Original">
</p>

**Model 1**

<p align="center">
   <img src="imgs/img_model1.png" alt="Model 1">
</p>

**Model 2**

<p align="center">
   <img src="imgs/img_model2.png" alt="Model 2">
</p>

**Model 3**

<p align="center">
   <img src="imgs/img_model3.png" alt="Model 3">
</p>

**Weighted Boxes Fusion**
<p align="center">
   <img src="imgs/img_wbf.png" alt="WBF Explanation">
</p>


## Weighed Boxes Fusion - Explanation

<p align="center">
   <img src="imgs/wbf.png" alt="WBF Explanation">
</p>

Stating in simple terms, WBF groups bounding boxes of i) IoUs over a threshold (0.55 in this paper) and ii) the same class. The grouped bounding boxes are considered as one object. The bounding boxes are adjusted by the weight of each bounding box prediction. This approach showed superior performance over Non-Maximum Supression (NMS) variants.

Check the paper for more : [arxiv: Weighted boxes fusion: Ensembling boxes from different object detection models](https://arxiv.org/pdf/1910.13302.pdf).

```
@article{solovyev2021weighted,
  title={Weighted boxes fusion: Ensembling boxes from different object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={Image and Vision Computing},
  pages={1-6},
  year={2021},
  publisher={Elsevier}
}
```
