#############################
### CREATED BY SUMIN HU #####
#############################

import torch, torchvision
from torch.nn.functional import threshold
# import some common libraries
import cv2, random, json, os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np
# check pytorch installation: 
print(torch.__version__, torch.cuda.is_available())
print(torchvision.__version__)
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# my libraries
from my_inference import sumin_inference_on_dataset
from ensemble_models import group_and_choose, Group_And_Choose, ConcatModels, LowScoreRemove, ErroneousClasses
from utils_sumin import cost_temp_2objects


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

class Eval:
    def __init__(self):
        print("\n\nMODEL 1: faster_rcnn_R_50_DC5_3x\n")
        self.cfg1 = get_cfg()
        self.cfg1.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"))
        self.cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg1.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml")
        self.predictor1 = DefaultPredictor(self.cfg1)
        print("\n\nMODEL 2: retinanet_R_101_FPN_3x\n")
        self.cfg2 = get_cfg()
        self.cfg2.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
        self.cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
        self.predictor2 = DefaultPredictor(self.cfg2)
        print("\n\nMODEL 3: faster_rcnn_X_101_32x8d_FPN_3x\n")
        self.cfg3 = get_cfg()
        self.cfg3.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg3.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg3.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor3 = DefaultPredictor(self.cfg3)

    def evaluate_pretrained(self):

        #################### MODEL 1: faster_rcnn_R_50_DC5_3x ###############################
        print("\n\nMODEL 1: faster_rcnn_R_50_DC5_3x\n")
        evaluator1 = COCOEvaluator("coco_2017_val", output_dir="./output1/")
        val_loader1 = build_detection_test_loader(self.cfg1, "coco_2017_val")
        inference_on_dataset(self.predictor1.model, val_loader1, evaluator1)

        #################### MODEL 2: retinanet_R_101_FPN_3x ###############################
        print("\n\nMODEL 2: retinanet_R_101_FPN_3x\n")
        evaluator2 = COCOEvaluator("coco_2017_val", output_dir="./output2/")
        val_loader2 = build_detection_test_loader(self.cfg2, "coco_2017_val")
        inference_on_dataset(self.predictor2.model, val_loader2, evaluator2)

        #################### MODEL 3: faster_rcnn_X_101_32x8d_FPN_3x ###############################
        print("\n\nMODEL 3: faster_rcnn_X_101_32x8d_FPN_3x\n")
        evaluator3 = COCOEvaluator("coco_2017_val", output_dir="./output3/")
        val_loader3 = build_detection_test_loader(self.cfg3, "coco_2017_val")
        inference_on_dataset(self.predictor3.model, val_loader3, evaluator3)

    def evaluate_one(self, image_id):
        # evalaute and visualize for one image
        print(f"Evaluating for image {image_id}")
        pth = os.path.join("datasets/coco/val2017",image_id)
        if not os.path.exists(pth):
            raise ValueError(f"image [{pth}] doesn't exist! Check the image id again")
        im = cv2.imread(pth)
        cv2.imshow("original image", im)
        out1 = self.predictor1(im)
        out2 = self.predictor2(im)
        out3 = self.predictor3(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg1.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(out1["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model 1",out.get_image()[:, :, ::-1])

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg2.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(out2["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model 2",out.get_image()[:, :, ::-1])

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg3.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(out3["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model 3",out.get_image()[:, :, ::-1])

        cat = Instances.cat([out1['instances'],out2['instances'],out3['instances']])
        ret = {}
        ret['instances'] = cat
        out = v.draw_instance_predictions(ret["instances"].to("cpu"))
        cv2.imshow("model SUM",out.get_image()[:, :, ::-1])

        while(True):
            k = cv2.waitKey(0)
            if k==27: #ESC
                break
    
    def concat_models_test(self):
        #################### MODEL Concatenation ###############################
        print("\nMODEL Concatenation\n")
        evaluator = COCOEvaluator("coco_2017_val", output_dir="./output_model/")
        val_loader = build_detection_test_loader(self.cfg3, "coco_2017_val")
        concat_predictor = ConcatModels(self.predictor1,self.predictor2,self.predictor3)
        sumin_inference_on_dataset(concat_predictor, val_loader, evaluator)
        print("FINISH GROUP & CHOOSE")

    def remove_low_scores(self,score_threshold):
        print("using model 2 to reduce FP")
        evaluator = COCOEvaluator("coco_2017_val",output_dir="./output_dump")
        val_loader = build_detection_test_loader(self.cfg3, "coco_2017_val")
        remove_low_predictor = LowScoreRemove(self.predictor1,self.predictor2,self.predictor3, score_threshold)
        sumin_inference_on_dataset(remove_low_predictor, val_loader, evaluator)

    def erroneous_classes(self):
        wrong_classifier_predictor = ErroneousClasses(self.predictor1)
        evaluator = COCOEvaluator("coco_2017_val",output_dir="./output_dump")
        val_loader = build_detection_test_loader(self.cfg1, "coco_2017_val")
        sumin_inference_on_dataset(wrong_classifier_predictor, val_loader, evaluator)

    def remove_low_scores_one(self, image_id, score_threshold):
        # evalaute and visualize for one image
        print(f"Evaluating for image {image_id}")
        pth = os.path.join("datasets/coco/val2017",image_id)
        if not os.path.exists(pth):
            raise ValueError(f"image [{pth}] doesn't exist! Check the image id again")
        im = cv2.imread(pth)
        print(f"this is the image we test for")
        cv2.imshow("original image", im)

        catModel = ConcatModels(self.predictor1,self.predictor2,self.predictor3)
        outputs = catModel(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg3.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model Concat",out.get_image()[:, :, ::-1])

        remove_low_predictor = LowScoreRemove(self.predictor1,self.predictor2,self.predictor3, score_threshold)
        outputs = remove_low_predictor(im)
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg3.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        cv2.imshow("model RemoveLowScore",out.get_image()[:, :, ::-1])

        while(True):
            k = cv2.waitKey(0)
            if k==27: #ESC
                break

    def grouping_one(self, image_id):
        # evalaute and visualize for one image
        print(f"Evaluating for image {image_id}")
        pth = os.path.join("datasets/coco/val2017",image_id)
        if not os.path.exists(pth):
            raise ValueError(f"image [{pth}] doesn't exist! Check the image id again")
        im = cv2.imread(pth)
        cv2.imshow("original image", im)

        out1 = self.predictor1(im)
        out2 = self.predictor2(im)
        out3 = self.predictor3(im)

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg1.DATASETS.TRAIN[0]), scale=1.2)
        im1 = v.draw_instance_predictions(out1["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model 1",im1.get_image()[:, :, ::-1])

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg2.DATASETS.TRAIN[0]), scale=1.2)
        im2 = v.draw_instance_predictions(out2["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model 2",im2.get_image()[:, :, ::-1])

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg3.DATASETS.TRAIN[0]), scale=1.2)
        im3 = v.draw_instance_predictions(out3["instances"].to("cpu"))
        # plt.imshow(out.get_image()[:, :, ::-1])
        cv2.imshow("model 3",im3.get_image()[:, :, ::-1])

        out_sum = group_and_choose([out1],[out2],[out3],threshold=0.5)
        im_final = v.draw_instance_predictions(out_sum[0]["instances"].to("cpu"))
        cv2.imshow("model SUM",im_final.get_image()[:, :, ::-1])
        while(True):
            k = cv2.waitKey(0)
            if k==27: # ESC
                break
    
    def groupNchoose_test(self):
        #################### MODEL Group And Choose ###############################
        print("\nMODEL Group And Choose\n")
        evaluator = COCOEvaluator("coco_2017_val", output_dir="./output_model/")
        val_loader = build_detection_test_loader(self.cfg3, "coco_2017_val")
        gnc = Group_And_Choose(self.predictor1,self.predictor2,self.predictor3)
        sumin_inference_on_dataset(gnc, val_loader, evaluator)
        print("FINISH GROUP & CHOOSE")

#### MAIN ####
def main():
    """
    SET MODE 
    * mode = 1    : Get AP result for Model 1,2,3
    * mode = 2    : Concatenate Models Test
    * mode = 2.1  : Visual Comparison of Model 1,2,3, and Concatenated Model
    * mode = 3    : score 낮은 bbox 없애는 model (FP를 줄이는 방안)
    * mode = 3.1  : Classification이 모두 틀리면 AP에 영향을 끼치는가? YES!
    * mode = 3.2  : score 낮은 bbox 없애기 for 1 image and visualize 
    * mode = 4    : Group and Choose Model Test
    * mode = 4.1  : Group and Choose single image test & visualization
    """
    mode = 4.1 # EDIT HERE
    print(f"######################\nMODE IS [{mode}]!!\n######################")

    eval = Eval()
    if mode==1:
        eval.evaluate_pretrained()
    elif mode==2:
        eval.concat_models_test()        
    elif mode==2.1:
        eval.evaluate_one("000000439715.jpg")
    elif mode==3:
        score_threshold = 0.5
        eval.remove_low_scores(score_threshold)
    elif mode==3.1:
        eval.erroneous_classes()
    elif mode==3.2:
        score_threshold = 0.5
        eval.remove_low_scores_one("000000439715.jpg",score_threshold)
    elif mode==4:
        eval.groupNchoose_test()
    elif mode==4.1:
        eval.grouping_one("000000439715.jpg")
    else:
        print("CHOOSE A MODE TO RUN!")


if __name__ == "__main__":
    main()









        # x : input image (w, h, c)
        # feature_map: w', h', c'
        # bbox: (x1, y1, x2, y2)

        # feature_map1 = self.predictor1.model.backbone(x) 
        # feature_map2 = self.predictor2.model.backbone(x) 
        # feature_map3 = self.predictor3.model.backbone(x) 
        # feature_map1 != feature_map2

        # bboxes = self.predictor1.model(x)

        # roi_allign = RoIAllign(height, width) (height, width: hyperparameters, maskr-cnn 7x7)

        # predictor1_list = []
        
        # # Reshape bbox to feat spatial dimension



        # for bbox in bboxes:
        #     bbox_feat = roi_allign(featuremap, bbox, box_index) => (c, 7, 7)
        #     predictor1_list.append((box_index, bbox_feat))

        # for feat1 in predictor1_list:
        #     for feat2 in predictor2_list:
        #         feat[1] = argmin_i{L2(feat1, feat2_i)}


        
