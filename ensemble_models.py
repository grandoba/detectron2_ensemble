import torch, torchvision
from detectron2.structures import Instances
from utils_sumin import xyxy_IoU
import numpy as np

# def my_model(input1, input2, input3):
#     cat = Instances.cat([input1['instances'],input2['instances'],input3['instances']])
#     ret = {}
#     ret['instances'] = cat
#     return ret

# def model_reduce_FP(input,thresh):    
#     z = input['instances'].scores > thresh 
#     input['instances'] = input['instances'][z]
#     return input

# def erroneous_class(input):
#     z = torch.empty(input[0]['instances'].pred_classes.size())
#     z[:] = 5
#     z = z.to('cuda:0')
#     input[0]['instances'].pred_classes = z
#     return input


def add_to_group(group, i,j) -> bool:
    found = False
    if len(group) == 0:
        group.append([i,j])
        return True
    for part in group:
        if i in part:
            part.append(j)
            found = True
            return True
        if j in part:
            part.append(i)
            found = True
            return True
    if not found:
        group.append([i,j])
        return True
    return False

def choose_one_from(all_instances, parts) -> Instances:
    """
        INPUT 1 - parts: list(int) 
            chosen instances' idx that are of the same class and have a high IoU
        INPUT 2 - all_instances: Instances 
            a concatenated list of instances
        OUTPUT - ret:
            one instance that is the best of the grouped instances

    """
    scores = [all_instances[i].scores.cpu().numpy().item(0) for i in parts]
    parts_of_instances = [all_instances[i] for i in parts]
    order = np.argsort(scores)
    pick_last = parts_of_instances[order[-1]]
    return pick_last


def group_and_choose(out1:list,out2:list,out3:list, threshold=0.5, lower_threshold=0) -> list:
    """
        out의 format: list({dict})
        out1[0]["instances"] : Instances
        out2[0]["instances"] : Instances
        out3[0]["instances"] : Instances
        threshold : float
            IoU > threshold will be considered as matched bboxes 
        lower_threshold: float
            remove bboxes below this threshold (0 means not removing any bboxes)

    """
    # Group detected bboxes by IoU >= threshold and same class
    all_instances:Instances = Instances.cat([out1[0]['instances'],out2[0]['instances'],out3[0]['instances']])
    N = len(all_instances)
    # list_bbox = [all_instances[i] for i in range(N)]
    group = []
    for i in range(N):
        for j in range(i+1,N):
            if all_instances[i].pred_classes != all_instances[j].pred_classes:
                continue
            iou = xyxy_IoU(all_instances[i].pred_boxes.tensor.cpu().numpy(),all_instances[j].pred_boxes.tensor.cpu().numpy())
            if iou >= threshold: # a match
                if not add_to_group(group, i,j):
                    raise ValueError(f"adding [{i}] and [{j}] to a group went wrong")
                # print("a pair grouped successfully")
    
    ## Choose best one out of the grouped bboxes
    instance_list = []
    for parts in group:
        chosen = choose_one_from(all_instances,parts)
        instance_list.append(chosen)
    
    ## Amongst the non grouped bboxes, choose high score boxes
    all_grouped = []
    for parts in group:
        for part in parts:
            all_grouped.append(part)
    all_non_grouped = []
    for i in range(N):
        if not i in all_grouped:
            all_non_grouped.append(i)
    scores = np.array([all_instances[i].scores.cpu().numpy().item(0) for i in all_non_grouped])
    parts_of_instances = [all_instances[i] for i in all_non_grouped]
    trueORfalse = scores > lower_threshold
    for idx, tof in enumerate(trueORfalse):
        if tof:
            instance_list.append(parts_of_instances[idx])

    if len(instance_list) == 0:
        instance_list.append(parts_of_instances[0])
    
    final = Instances.cat(instance_list)
    ret = {}
    ret['instances'] = final
    return [ret]


class Group_And_Choose(object):
    def __init__(self, predictor1, predictor2, predictor3, thr1=0.5, thr2=0):
        self.predictors = []
        self.predictors.append(predictor1)
        self.predictors.append(predictor2)
        self.predictors.append(predictor3)
        self.threshold = thr1
        self.lower_threshold = thr2

    def __call__(self,inputs) -> dict:
        out1 = self.predictors[0].model(inputs)
        out2 = self.predictors[1].model(inputs)
        out3 = self.predictors[2].model(inputs)
        output = group_and_choose(out1,out2,out3, self.threshold, self.lower_threshold)
        return output

class ConcatModels(object):
    def __init__(self, predictor1, predictor2, predictor3):
        self.predictors = []
        self.predictors.append(predictor1)
        self.predictors.append(predictor2)
        self.predictors.append(predictor3)

    def __call__(self,inputs) -> dict:
        out1 = self.predictors[0].model(inputs)
        out2 = self.predictors[1].model(inputs)
        out3 = self.predictors[2].model(inputs)
        final = Instances.cat([out1[0]["instances"],out2[0]["instances"],out3[0]["instances"]])
        ret = {}
        ret['instances'] = final
        return [ret]

class LowScoreRemove(object):
    def __init__(self, predictor1, predictor2, predictor3,threshold):
        self.predictors = []
        self.predictors.append(predictor1)
        self.predictors.append(predictor2)
        self.predictors.append(predictor3)
        self.thrs = threshold

    def __call__(self,inputs) -> dict:
        out1 = self.predictors[0].model(inputs)
        out2 = self.predictors[1].model(inputs)
        out3 = self.predictors[2].model(inputs)
        final = Instances.cat([out1[0]["instances"],out2[0]["instances"],out3[0]["instances"]])
        chosen = final.scores > self.thrs
        final = final[chosen]
        ret = {}
        ret['instances'] = final
        return [ret]

class ErroneousClasses(object):
    def __init__(self, predictor):
        self.predictor = predictor

    def __call__(self,inputs) -> dict:
        outputs = self.predictor.model(inputs)
        z = torch.empty(outputs[0]['instances'].pred_classes.size())
        z[:] = 0
        z = z.to('cuda:0')
        outputs[0]['instances'].pred_classes = z
        return outputs

# class WBFmodel(object):
#     def __init__(self, predictions_per_image: list):
#         self.predictions = predictions_per_image
#     def __call__(self,inputs) -> dict:
#         outputs = 
#         return outputs