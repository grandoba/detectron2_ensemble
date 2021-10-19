# detectron2_ensemble

## Prerequisites
0. Basics
```sh
# Uses Python3, Numpy, Numba
```

1. [Detectron2 (:=pytorch framework)](https://github.com/facebookresearch/detectron2)
```sh
# Install by the link below
# https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

2. [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
```sh
pip3 install ensemble-boxes # or just pip 
```

3. Git clone this repository
```sh
git clone git@github.com:grandoba/detectron2_ensemble.git
```

4. Download the COCO datasets [ref](https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9)
```sh
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

# Need?
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip
unzip image_info_unlabeled2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip

```

## Reproducing the results
```sh
# first run mode=1 to get the results from each model and save to directory "outputs1", "outputs2", and "outputs3"
python3 ensemble.py --mode 1

# Run WBF ensemble: AP@[0.50:0.95] = 43.30
python3 ensemble.py --mode 5
```





