#############################
### CREATED BY SUMIN HU #####
#############################

import torch, torchvision
from torch.nn.functional import threshold
# check pytorch installation: 
print(torch.__version__, torch.cuda.is_available())
print(torchvision.__version__)
assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

# import some common libraries
import cv2, random, json, os, argparse, tqdm
import numpy as np
from collections import defaultdict


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
from detectron2.structures import Instances, Boxes, BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# import ensemble_boxes
from ensemble_boxes import *

# my libraries
from my_inference import sumin_inference_on_dataset
from ensemble_models import *


##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

class Eval:
    def __init__(self, model1_yaml="faster_rcnn_R_50_DC5_3x.yaml", model2_yaml="retinanet_R_101_FPN_3x.yaml", model3_yaml="faster_rcnn_X_101_32x8d_FPN_3x.yaml"):
        
        print(f"\n\nMODEL 1: {model1_yaml}\n")
        yaml_path1 = os.path.join("COCO-Detection",model1_yaml)
        self.cfg1 = get_cfg()
        self.cfg1.merge_from_file(model_zoo.get_config_file(yaml_path1))
        self.cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg1.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_path1)
        self.predictor1 = DefaultPredictor(self.cfg1)

        print(f"\n\nMODEL 2: {model2_yaml}\n")
        yaml_path2 = os.path.join("COCO-Detection",model2_yaml)
        self.cfg2 = get_cfg()
        self.cfg2.merge_from_file(model_zoo.get_config_file(yaml_path2))
        self.cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_path2)
        self.predictor2 = DefaultPredictor(self.cfg2)

        print(f"\n\nMODEL 3: {model3_yaml}\n")
        yaml_path3 = os.path.join("COCO-Detection",model3_yaml)
        self.cfg3 = get_cfg()
        self.cfg3.merge_from_file(model_zoo.get_config_file(yaml_path3))
        self.cfg3.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg3.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_path3)
        self.predictor3 = DefaultPredictor(self.cfg3)

        # Make sure the output directories are available
        dirs = ["output1", "output2", "output3", "output_model", "output_dump"]
        for dir in dirs:
            if not os.path.exists(dir):
                os.mkdir(dir)

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
        gnc = Group_And_Choose(self.predictor1,self.predictor2,self.predictor3,0.5,0.7)
        sumin_inference_on_dataset(gnc, val_loader, evaluator)
        print("FINISH GROUP & CHOOSE")
    
    def WBF_one(self, image_id, skip_thr=0):
        """
            image_id    : string    (e.g. "000000439715.jpg")
            skip_thr    : float     (bbox of scores below this threshold will not be used for WBF)
        """
        # evalaute and visualize for one image
        print(f"Evaluating for image {image_id}")
        pth = os.path.join("datasets/coco/val2017",image_id)
        if not os.path.exists(pth):
            raise ValueError(f"image [{pth}] doesn't exist! Check the image id again")
        img = cv2.imread(pth)
        cv2.imshow("original image", img)


        ## STEP 0: FIND ALL BBOXES IN IMAGE image_id: read "coco_instances_results.json"
        
        f = open("output1/coco_instances_results.json")
        json1 = json.load(f)
        f = open("output2/coco_instances_results.json")
        json2 = json.load(f)
        f = open("output3/coco_instances_results.json")
        json3 = json.load(f)
        pred_by_model1  = defaultdict(list)
        pred_by_model2  = defaultdict(list)
        pred_by_model3  = defaultdict(list)
        for p in json1:
            pred_by_model1[p["image_id"]].append(p)
        for p in json2:
            pred_by_model2[p["image_id"]].append(p)
        for p in json3:
            pred_by_model3[p["image_id"]].append(p)

        dicts = list(DatasetCatalog.get("coco_2017_val"))
        metadata = MetadataCatalog.get("coco_2017_val")
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            def dataset_id_map(ds_id):
                return metadata.thing_dataset_id_to_contiguous_id[ds_id]

        for dic in tqdm.tqdm(dicts):
            if dic["file_name"][-16:] != image_id:
                continue
            # img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
            img_shp = img.shape[:2] # tuple(height,width )
            predictions1 = pred_by_model1[dic["image_id"]]
            predictions2 = pred_by_model2[dic["image_id"]]
            predictions3 = pred_by_model3[dic["image_id"]]

        ## STEP 2: PRE-PROCESSING FOR weighted_box_fusion FUNCTION

            ## step 2-A: get boxes_list, scores_list, labels_list
            scores1 = np.asarray([x['score'] for x in predictions1])
            chosen1 = (scores1 > skip_thr ).nonzero()[0]
            scores1 = scores1[chosen1]
            bbox1 = np.asarray([predictions1[i]['bbox'] for i in chosen1]).reshape(-1,4)
            bbox1 = BoxMode.convert(bbox1, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            new_box1 = bbox1.copy()
            new_box1[:,[1,3]] = new_box1[:,[1,3]]/img_shp[0] #normalize y by height
            new_box1[:,[0,2]] = new_box1[:,[0,2]]/img_shp[1] #normalize x by width
            labels1 = np.asarray([dataset_id_map(predictions1[i]['category_id']) for i in chosen1])

            scores2 = np.asarray([x['score'] for x in predictions2])
            chosen2 = (scores2 > skip_thr ).nonzero()[0]
            scores2 = scores2[chosen2]
            bbox2 = np.asarray([predictions2[i]['bbox'] for i in chosen2]).reshape(-1,4)
            bbox2 = BoxMode.convert(bbox2, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            new_box2 = bbox2.copy()
            new_box2[:,[1,3]] = new_box2[:,[1,3]]/img_shp[0] #normalize y by height
            new_box2[:,[0,2]] = new_box2[:,[0,2]]/img_shp[1] #normalize x by width
            labels2 = np.asarray([dataset_id_map(predictions2[i]['category_id']) for i in chosen2])

            scores3 = np.asarray([x['score'] for x in predictions3])
            chosen3 = (scores3 > skip_thr ).nonzero()[0]
            scores3 = scores3[chosen3]
            bbox3 = np.asarray([predictions3[i]['bbox'] for i in chosen3]).reshape(-1,4)
            bbox3 = BoxMode.convert(bbox3, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            new_box3 = bbox3.copy()
            new_box3[:,[1,3]] = new_box3[:,[1,3]]/img_shp[0] #normalize y by height
            new_box3[:,[0,2]] = new_box3[:,[0,2]]/img_shp[1] #normalize x by width
            labels3 = np.asarray([dataset_id_map(predictions3[i]['category_id']) for i in chosen3])
            
        ## STEP 3: weighted_box_fusion APPLY
            weights = [1,1,1] # CHANGE IT LATER ACCORDING TO TOTAL AP SCORE PROPORTIONS
            boxes_list = [new_box1.tolist(), new_box2.tolist(), new_box3.tolist()]
            scores_list = [scores1.tolist(), scores2.tolist(), scores3.tolist()]
            labels_list = [labels1.tolist(), labels2.tolist(), labels3.tolist()]
            final_boxes, final_scores, final_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights,iou_thr=0.5,skip_box_thr=skip_thr)
            final_boxes[:,[1,3]] = final_boxes[:,[1,3]]*img_shp[0] # restore y by height
            final_boxes[:,[0,2]] = final_boxes[:,[0,2]]*img_shp[1] # restore x by width
        ## STEP 4: VISUALIZE RESULT
            ## STEP 4-A: make model 1, 2, 3, wbf into instances
            ret1 = Instances(img_shp)
            ret1.scores = scores1
            ret1.pred_boxes = Boxes(bbox1)
            ret1.pred_classes = labels1

            ret2 = Instances(img_shp)
            ret2.scores = scores2
            ret2.pred_boxes = Boxes(bbox2)
            ret2.pred_classes = labels2

            ret3 = Instances(img_shp)
            ret3.scores = scores3
            ret3.pred_boxes = Boxes(bbox3)
            ret3.pred_classes = labels3

            ret_final = Instances(img_shp)
            ret_final.scores = np.asarray(final_scores)
            ret_final.pred_boxes = Boxes(final_boxes)
            ret_final.pred_classes = np.asarray(final_labels).astype(np.int64)


            ## STEP 4-B: Visualize output 1, 2, 3, and wbf

            v = Visualizer(img, metadata, scale=1.2)
            v_pred = v.draw_instance_predictions(ret1).get_image()
            v = Visualizer(img, metadata, scale=1.2)
            v_gt = v.draw_dataset_dict(dic).get_image()
            concat = np.concatenate((v_pred,v_gt), axis=1)
            cv2.imshow("model 1: pred & gt", concat[:, :, ::-1])

            v = Visualizer(img, metadata, scale=1.2)
            v_pred = v.draw_instance_predictions(ret2).get_image()
            v = Visualizer(img, metadata, scale=1.2)
            v_gt = v.draw_dataset_dict(dic).get_image()
            concat = np.concatenate((v_pred,v_gt), axis=1)
            cv2.imshow("model 2: pred & gt", concat[:, :, ::-1])

            v = Visualizer(img, metadata, scale=1.2)
            v_pred = v.draw_instance_predictions(ret3).get_image()
            v = Visualizer(img, metadata, scale=1.2)
            v_gt = v.draw_dataset_dict(dic).get_image()
            concat = np.concatenate((v_pred,v_gt), axis=1)
            cv2.imshow("model 3: pred & gt", concat[:, :, ::-1])
            
            v = Visualizer(img, metadata, scale=1.2)
            v_pred = v.draw_instance_predictions(ret_final).get_image()
            v = Visualizer(img, metadata, scale=1.2)
            v_gt = v.draw_dataset_dict(dic).get_image()
            concat = np.concatenate((v_pred,v_gt), axis=1)
            cv2.imshow("model Weight Box Fusion: pred & gt", concat[:, :, ::-1])

            while(True):
                k = cv2.waitKey(0)
                if k==27: # ESC
                    break
            
    def WBF_test(self, skip_thr=0.0):
        """
            image_id    : string    (e.g. "000000439715.jpg")
            skip_thr    : float     (bbox of scores below this threshold will not be used for WBF)
        """
        evaluator = COCOEvaluator("coco_2017_val",output_dir="./output_model")
        evaluator.reset()

        ## STEP 0: FIND ALL BBOXES -> read "coco_instances_results.json"
        f = open("output1/coco_instances_results.json")
        json1 = json.load(f)
        f = open("output2/coco_instances_results.json")
        json2 = json.load(f)
        f = open("output3/coco_instances_results.json")
        json3 = json.load(f)
        pred_by_model1  = defaultdict(list)
        pred_by_model2  = defaultdict(list)
        pred_by_model3  = defaultdict(list)
        for p in json1:
            pred_by_model1[p["image_id"]].append(p)
        for p in json2:
            pred_by_model2[p["image_id"]].append(p)
        for p in json3:
            pred_by_model3[p["image_id"]].append(p)

        dicts = list(DatasetCatalog.get("coco_2017_val"))
        metadata = MetadataCatalog.get("coco_2017_val")
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            def dataset_id_map(ds_id):
                return metadata.thing_dataset_id_to_contiguous_id[ds_id]

        for dic in tqdm.tqdm(dicts):
            img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
            img_shp = img.shape[:2] # tuple(height,width )
            predictions1 = pred_by_model1[dic["image_id"]]
            predictions2 = pred_by_model2[dic["image_id"]]
            predictions3 = pred_by_model3[dic["image_id"]]

        ## STEP 2: PRE-PROCESSING FOR weighted_box_fusion FUNCTION

            ## step 2-A: get boxes_list, scores_list, labels_list
            scores1 = np.asarray([x['score'] for x in predictions1])
            chosen1 = (scores1 > skip_thr ).nonzero()[0]
            scores1 = scores1[chosen1]
            bbox1 = np.asarray([predictions1[i]['bbox'] for i in chosen1]).reshape(-1,4)
            bbox1 = BoxMode.convert(bbox1, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            new_box1 = bbox1.copy()
            new_box1[:,[1,3]] = new_box1[:,[1,3]]/img_shp[0] #normalize y by height
            new_box1[:,[0,2]] = new_box1[:,[0,2]]/img_shp[1] #normalize x by width
            labels1 = np.asarray([dataset_id_map(predictions1[i]['category_id']) for i in chosen1])

            scores2 = np.asarray([x['score'] for x in predictions2])
            chosen2 = (scores2 > skip_thr ).nonzero()[0]
            scores2 = scores2[chosen2]
            bbox2 = np.asarray([predictions2[i]['bbox'] for i in chosen2]).reshape(-1,4)
            bbox2 = BoxMode.convert(bbox2, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            new_box2 = bbox2.copy()
            new_box2[:,[1,3]] = new_box2[:,[1,3]]/img_shp[0] #normalize y by height
            new_box2[:,[0,2]] = new_box2[:,[0,2]]/img_shp[1] #normalize x by width
            labels2 = np.asarray([dataset_id_map(predictions2[i]['category_id']) for i in chosen2])

            scores3 = np.asarray([x['score'] for x in predictions3])
            chosen3 = (scores3 > skip_thr ).nonzero()[0]
            scores3 = scores3[chosen3]
            bbox3 = np.asarray([predictions3[i]['bbox'] for i in chosen3]).reshape(-1,4)
            bbox3 = BoxMode.convert(bbox3, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            new_box3 = bbox3.copy()
            new_box3[:,[1,3]] = new_box3[:,[1,3]]/img_shp[0] #normalize y by height
            new_box3[:,[0,2]] = new_box3[:,[0,2]]/img_shp[1] #normalize x by width
            labels3 = np.asarray([dataset_id_map(predictions3[i]['category_id']) for i in chosen3])
            
        ## STEP 3: weighted_box_fusion APPLY
            weights = [1,1,1] # CHANGE IT LATER ACCORDING TO TOTAL AP SCORE PROPORTIONS
            boxes_list = [new_box1.tolist(), new_box2.tolist(), new_box3.tolist()]
            scores_list = [scores1.tolist(), scores2.tolist(), scores3.tolist()]
            labels_list = [labels1.tolist(), labels2.tolist(), labels3.tolist()]
            final_boxes, final_scores, final_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights,iou_thr=0.5,skip_box_thr=skip_thr)
            final_boxes[:,[1,3]] = final_boxes[:,[1,3]]*img_shp[0] # restore y by height
            final_boxes[:,[0,2]] = final_boxes[:,[0,2]]*img_shp[1] # restore x by width

        ## STEP 4: Append I/O into Evaluator
            ## STEP 4-A: make wbf output into instances
            ret_final = Instances(img_shp)
            ret_final.scores = np.asarray(final_scores)
            ret_final.pred_boxes = Boxes(final_boxes)
            ret_final.pred_classes = np.asarray(final_labels).astype(np.int64)
            outputs = {}
            outputs["instances"] = ret_final

            ## STEP 4-B: Process each input / output by evaluator.process(in,out)
            evaluator.process([dic],[outputs])

        print("\n\nNOW EVALUATING Average Percision\n")
        evaluator.evaluate()


    

#### MAIN ####
def main(mode:int):

    eval = Eval()
    """
        Default settings
            + model1_yaml="faster_rcnn_R_50_DC5_3x.yaml"
            + model2_yaml="retinanet_R_101_FPN_3x.yaml" 
            + model3_yaml="faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            
    """
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
    elif mode==5:
        eval.WBF_test(skip_thr=0.0)    
    elif mode==5.1:
        eval.WBF_one("000000439715.jpg", skip_thr=0.0)
    else:
        print(f"You chose mode |{mode}| that doesn't exist. Refer to the bottom of this script.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that runs ensembles on the COCO dataset."
    )
    parser.add_argument("--mode", required=True, help="Explanation of modes are at the bottom of the script")
    args = parser.parse_args()
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
    * mode = 5  : Weighted Box Fusion Model Test
    * mode = 5.1  : Weighted Box Fusion single image test & visualization
    """
    print(f"#######################\n##### MODE IS [{args.mode}] #####\n#######################\n")
    print("CAUTION: Must run mode==1 if running the first time.\n############################################")
    if not os.path.exists("datasets"):
        raise ValueError("You Need To Have 'datasets' directory in the working directory! Check out the Prerequisite section from 'README.md' file")

    main(mode=int(args.mode))


