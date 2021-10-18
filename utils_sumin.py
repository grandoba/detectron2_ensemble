import numpy as np
import scipy

def xyxy_IoU(bbox1:np.array, bbox2:np.array) -> float:
    """
        bbox1:  array([[114.0343 , 265.19604, 150.48466, 396.58865]], dtype=float32)
        bbox2:  array([[258.6244 , 161.54675, 337.15756, 407.74356]], dtype=float32)

        Caution: bbox1, bbox2 are 2D nd.arrays of shape (1,4)

    """    
    x1, y1, x2, y2 = bbox1[0]
    x3, y3, x4, y4 = bbox2[0]
    w1 = x2-x1
    w2 = x4-x3
    h1 = y2-y1
    h2 = y4-y3
    xx1 = max(x1,x3)
    yy1 = max(y1,y3)
    xx2 = min(x2,x4)
    yy2 = min(y2,y4)
    ww = max(0, xx2-xx1)
    hh = max(0, yy2-yy1)

    inter_area = ww*hh
    union_area = w1*h1 + w2*h2 - inter_area
    iou = inter_area/union_area
    return iou

def cost_temp_2objects(obj1:np.array, obj2:np.array) -> float:
    """
        obj1: [132.9278, 246.6379, 463.5482, 480.0000]
        obj2: [132.9278, 246.6379, 463.5482, 480.0000]

    """
    iou = xyxy_IoU(obj1,obj2)

    if iou<0.3:
        cost = 2
    else:
        cost = 1-iou
    return cost

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

def new_bbox(obj1, obj2, obj3):
    """
        INPUT: obj1, obj2, obj3
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

def compare_3images(image1,image2,image3):
    cat = Instances.cat([image1['instances'],image2['instances'],image3['instances']])
    ret = {}
    ret['instances'] = cat
    return ret

    cost1: np.array = make_cost_matrix_inst(image1,image2)
    cost2: np.array = make_cost_matrix_inst(image2,image3)
    cost3: np.array = make_cost_matrix_inst(image3,image1)

    ### Hungarian Alg: Assign One to One bbox
    row_idx1, col_idx1 = linear_sum_assignment(cost1)
    print(f"total cost is {cost1[row_idx1, col_idx1].sum()}")
    print(f"# of detected objects\nimage 1:{len(row_idx1)}\nimage 2:{len(col_idx1)}") 
    row_idx2, col_idx2 = linear_sum_assignment(cost2)
    print(f"total cost is {cost2[row_idx2, col_idx2].sum()}")
    print(f"# of detected objects\nimage 2:{len(row_idx2)}\nimage 3:{len(col_idx2)}")
    row_idx3, col_idx3 = linear_sum_assignment(cost3)
    print(f"total cost is {cost3[row_idx3, col_idx3].sum()}")
    print(f"# of detected objects\nimage 3:{len(row_idx3)}\nimage 1:{len(col_idx3)}")
    print("One to One assignment complete")

    ### New Bounding Box generation
    
def make_cost_matrix(image1, image2) -> np.array: 
    """
        # {'image_id': 581781, 'instances': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...]}
        # [{'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9891513586044312}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9800368547439575}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9772728085517883}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9727432727813721}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9678618907928467}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9676752090454102}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9419427514076233}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9274771809577942}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9216170310974121}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8331450819969177}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8210079073905945}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8013830184936523}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.7981986999511719}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.7894580364227295}, ...]

        input : instances
    """
    if image1['image_id'] != image2['image_id']:
        raise ValueError("wrong image comaprison")

    cost_matrix = np.zeros((len(image1),len(image2)),dtype=float)
    for i in range(len(image1)):
        for j in range(len(image2)):
            cost_ij = cost_bw_2objects(image1[i], image2[j])
            cost_matrix[i][j] = cost_ij

    return cost_matrix
            
def make_cost_matrix_inst(instance1, instance2) -> np.array: 
    """
        # {'image_id': 581781, 'instances': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...]}
        # [{'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9891513586044312}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9800368547439575}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9772728085517883}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9727432727813721}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9678618907928467}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9676752090454102}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9419427514076233}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9274771809577942}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.9216170310974121}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8331450819969177}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8210079073905945}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.8013830184936523}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.7981986999511719}, {'image_id': 581781, 'category_id': 46, 'bbox': [...], 'score': 0.7894580364227295}, ...]

        ref: https://github.com/facebookresearch/detectron2/issues/603
        input : instances 
        # image1['instances'].pred_boxes.tensor.cpu().numpy()
        
        # image1['instances'] 
        Instances(num_instances=17, image_height=480, image_width=640, fields=[pred_boxes: Boxes(tensor([[132.9278, 246.6379, 463.5482, 480.0000],
        [258.6244, 161.5468, 337.1576, 407.7436],
        [  0.0000, 278.5832,  75.1529, 475.5381],
        [114.0343, 265.1960, 150.4847, 396.5887],
        [ 48.2085, 275.0251,  79.8162, 345.1213],
        [561.0758, 273.3667, 595.5221, 367.1072],
        [509.0049, 265.2365, 573.1155, 292.7913],
        [384.6914, 272.1828, 412.1364, 307.3944],
        [526.7502, 280.6893, 562.1566, 348.1790],
        [331.5553, 230.0806, 393.5180, 256.6008],
        [338.0782, 252.1636, 414.9969, 280.6153],
        [507.3170, 282.9673, 533.8842, 347.8902],
        [345.3246, 269.7372, 384.5261, 298.9379],
        [407.5568, 273.3968, 454.9272, 341.6323],
        [594.8682, 263.4063, 613.6784, 314.7442],
        [374.8466, 252.5861, 414.1368, 277.7503],
        [552.7322, 257.5269, 572.4736, 306.0464]], device='cuda:0')), scores: tensor([0.9998, 0.9957, 0.9947, 0.9929, 0.9897, 0.9799, 0.9764, 0.9725, 0.9703,
        0.9622, 0.9571, 0.9174, 0.8942, 0.7984, 0.6981, 0.5772, 0.5713],
       device='cuda:0'), pred_classes: tensor([17,  0,  0,  0,  0,  0, 25,  0,  0, 25, 25,  0,  0,  0,  0, 25,  0],
       device='cuda:0')])

    """
    inst1:np.array = instance1['instances'][instance1['instances'].pred_classes!=0].pred_boxes.tensor.cpu().numpy()
    inst2:np.array = instance2['instances'][instance2['instances'].pred_classes!=0].pred_boxes.tensor.cpu().numpy()

    cost_matrix = np.zeros((len(inst1),len(inst2)),dtype=float)
    for i in range(len(inst1)):
        for j in range(len(inst2)):
            cost_ij = cost_temp_2objects(inst1[i], inst2[j])
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
    xx2 = min(x1+h1,x2+h2)
    yy2 = min(y1+w1,y2+w2)
    ww = max(0,xx2-xx1)
    hh = max(0,yy2-yy1)
    
    inter_area = ww*hh
    union_area = w1*h1 + w2*h2 - inter_area
    iou = inter_area/union_area
    if iou<0.3:
        cost = 2
    else:
        cost = 1-iou
    
    return cost
    # 일단 보류: calculate category detection => majority voting하는 방식으로? 
    # return (1-iou)*0.5 + category_cost*0.5



# {"image_id": 139, "category_id": 72, "bbox": [6.554643630981445, 167.0744171142578, 146.7380828857422, 95.85887145996094], "score": 0.9983035326004028}, 



# https://github.com/cocodataset/cocoapi/issues/102
# For object detection annotations, the format is "bbox" : [x,y,width,height]
# Where:
# x, y: the upper-left coordinates of the bounding box
# width, height: the dimensions of your bounding box