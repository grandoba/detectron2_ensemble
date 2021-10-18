# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
print(torchvision.__version__)
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# !wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
# im = cv2.imread("./input.jpg")
# plt.imshow(im)

def main():
    #################### MODEL 1: faster_rcnn_R_50_DC5_3x ###############################
    print("\n\nMODEL 1: faster_rcnn_R_50_DC5_3x\n")
    cfg1 = get_cfg()
    cfg1.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"))
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg1.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml")
    predictor = DefaultPredictor(cfg1)
    evaluator = COCOEvaluator("coco_2017_val", output_dir="./output1/")
    val_loader = build_detection_test_loader(cfg1, "coco_2017_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    
    #################### MODEL 2: retinanet_R_101_FPN_3x ###############################
    print("\n\nMODEL 2: retinanet_R_101_FPN_3x\n")
    cfg2 = get_cfg()
    cfg2.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg2)
    evaluator = COCOEvaluator("coco_2017_val", output_dir="./output2/")
    val_loader = build_detection_test_loader(cfg2, "coco_2017_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)


    #################### MODEL 3: faster_rcnn_X_101_32x8d_FPN_3x ###############################
    print("\n\nMODEL 3: faster_rcnn_X_101_32x8d_FPN_3x\n")
    cfg3 = get_cfg()
    cfg3.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg3.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg3.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg3)
    evaluator = COCOEvaluator("coco_2017_val", output_dir="./output3/")
    val_loader = build_detection_test_loader(cfg3, "coco_2017_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)

if __name__ == "__main__":
    main()