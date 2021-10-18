
# {"image_id": 139, "category_id": 72, "bbox": [6.554643630981445, 167.0744171142578, 146.7380828857422, 95.85887145996094], "score": 0.9983035326004028}, 



# https://github.com/cocodataset/cocoapi/issues/102
# For object detection annotations, the format is "bbox" : [x,y,width,height]
# Where:
# x, y: the upper-left coordinates of the bounding box
# width, height: the dimensions of your bounding box

import scipy
import torch, torchvision
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
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from my_inference import my_inference_on_dataset

##### METHOD 1 MAJORITY VOTING #####


def bbox_matching(data1, data2):
    """Bounding Box Matching problem solved using Hungarian alg.

    Parameters
    ----------
    data: dictionary
        the entire output from a detection model
    
    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.

    Notes
    -----
    The linear sum assignment problem [1]_ is also known as minimum weight
    matching in bipartite graphs. A problem instance is described by a matrix
    C, where each C[i,j] is the cost of matching vertex i of the first partite
    set (a "worker") and vertex j of the second set (a "job"). The goal is to
    find a complete assignment of workers to jobs of minimal cost.

    """

    # sort images in order so they have matching ids (just in case) (missing images may corrupt this process)
    img_id_4_model1 = [i['image_id'] for i in data1]
    img_id_4_model2 = [i['image_id'] for i in data2]
    idx1 = np.argsort(img_id_4_model1)
    idx2 = np.argsort(img_id_4_model2)

    for i in range(len(idx1)):
        ### Make Cost Matrix for images of ID: ㅁㅁㅁ
        image1 = data1[idx1[i]]
        image2 = data2[idx2[i]]
        cost_mat = make_cost_matrix(image1,image2)
        
        ### Hungarian Alg: Assign One to One bbox
        row_idx, col_idx = linear_sum_assignment(cost_mat)
        print(f"total cost is {cost_mat[row_idx, col_idx].sum()}")
        print(f"# of detected objects\nimage 1:{len(row_idx)}\nimage 2:{len(col_idx)}") 
        
        # return a new bbox and category (currently just choose 1 model randomly)
        for j in range(len(row_idx)):
            # takes in objects 1 and object 2 and return object 3
            new_obj = new_bbox(image1[row_idx[j]], image2[col_idx[j]])
            
        
    ###

def new_bbox(obj1, obj2):
    """
        INPUT: obj1, obj2
        {   
            "image_id": 139, 
            "category_id": 72, 
            "bbox": [6.554643630981445, 167.0744171142578, 146.7380828857422, 95.85887145996094], 
            "score": 0.9983035326004028
        }

    """
    if np.random.rand() > 0.5:
        return obj1.copy()
    else:
        return obj2.copy()

def make_cost_matrix(image1, image2): 
    """
        # {'image_id': 581781, 'instances': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...]}
        # [{'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9891513586044312}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9800368547439575}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9772728085517883}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9727432727813721}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9678618907928467}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9676752090454102}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9419427514076233}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9274771809577942}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9216170310974121}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8331450819969177}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8210079073905945}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8013830184936523}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.7981986999511719}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.7894580364227295}, ...]

        input : instances
    """
    if image1['image_id'] != image2['image_id']:
        raise RuntimeError("wrong image comaprison")

    cost_matrix = np.zeros((len(image1),len(image2)),dtype=float)
    for i in range(len(image1)):
        for j in range(len(image2)):
            cost_ij = cost_bw_2objects(image1[i], image2[j])
            cost_matrix[i][j] = cost_ij

    return cost_matrix
            

def cost_bw_2objects(obj1, obj2) -> float:
    """
        obj1: {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9891513586044312}
        obj2: {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9891513586044312}

    """
    # calcuate IoU
    x1, y1, w1, h1 = obj1['bbox']
    x2, y2, w2, h2 = obj2['bbox']
    
    xx1 = max(x1,x2)
    yy1 = max(y1,y2)
    xx2 = min(x1+h1,x1+h2)
    yy2 = min(y1+w1,y1+w2)
    ww = max(0,xx2-xx1)
    hh = max(0,yy2-yy1)
    
    inter_area = ww*hh
    union_area = w1*h1 + w2*h2 - inter_area
    iou = inter_area/union_area
    
    return 1-iou
    # 일단 보류: calculate category detection => majority voting하는 방식으로? 
    # return (1-iou)*0.5 + category_cost*0.5
    


# #### MAIN ####
# if __name__ == "__main__":
#     print("\n### ENSEMBLE METHOD 1: ###")
#     dir1 = "instances_predictions.pth"
#     path1 = os.path.join("output1",dir1)
#     if not os.path.exists(path1):
#         raise RuntimeError("{%s} not available. Check it again~", path1)
#     file1 = open(path1)
#     data1 = json.load(file1)
#     print("there are {} input images",len(data1))
#     for image in data1:
#         print(object['image_id'])
#         print(object['category_id'])
#         print(object['bbox'])
#         print(object['bbox'][0])
#         print(object['bbox'][1])
#         print(object['bbox'][2])
#         print(object['bbox'][3])
#         print(object['score'])
    


    

def main():

    models = []
    #################### MODEL 1: faster_rcnn_R_50_DC5_3x ###############################
    print("\n\nMODEL 1: faster_rcnn_R_50_DC5_3x\n")
    cfg1 = get_cfg()
    cfg1.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml"))
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg1.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml")
    predictor1 = DefaultPredictor(cfg1)
    models.append(predictor1.model)

    #################### MODEL 2: retinanet_R_101_FPN_3x ###############################
    print("\n\nMODEL 2: retinanet_R_101_FPN_3x\n")
    cfg2 = get_cfg()
    cfg2.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    predictor2 = DefaultPredictor(cfg2)
    models.append(predictor2.model)

    #################### MODEL 3: faster_rcnn_X_101_32x8d_FPN_3x ###############################
    print("\n\nMODEL 3: faster_rcnn_X_101_32x8d_FPN_3x\n")
    cfg3 = get_cfg()
    cfg3.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg3.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg3.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor3 = DefaultPredictor(cfg3)
    models.append(predictor3.model)

    evaluator = COCOEvaluator("coco_2017_val", output_dir="./output3/")
    val_loader = build_detection_test_loader(cfg3, "coco_2017_val")
    my_inference_on_dataset(models, val_loader, evaluator)

if __name__ == "__main__":
    main()