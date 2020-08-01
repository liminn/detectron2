import os
import cv2
import torch 
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class PnopticFPN(object):
    def __init__(self, ckpt_path="", min_size = 800, max_size = 1333):
        super(PnopticFPN, self).__init__()
        self.ckpt_path = ckpt_path
        self.min_size = min_size
        self.max_size = max_size
        self.device = "cuda"
    def _get_model(self, ckpt_path):
        model = torch.load(ckpt_path, map_location=self.device).to(self.device).eval()
        return model
    
    def _preprocess(self,image):
        ### control the max_size and min_size of image
        h_origin, w_origin, c = image.shape
        h, w, c = image.shape
        val_min_size = self.min_size
        val_max_size = self.max_size
        # 0.control min_size
        new_h, new_w = h, w
        if(w<val_min_size or h<val_min_size):
            if(w<h):
                new_w = val_min_size
                new_h = int(h/w*new_w)
            else:
                new_h = val_min_size
                new_w = int(w/h*new_h)
        h, w =  new_h, new_w 
        # 1.control max_size
        if(w>val_max_size or h>val_max_size):
            if(w>h):
                new_w = val_max_size
                new_h = int(h/w*new_w)
            else:
                new_h = val_max_size
                new_w = int(w/h*new_h)
        h, w = new_h, new_w
        # 2. %32==0
        new_h = h//32*32
        new_w = w//32*32
        # resize
        #print("origin shape:{}".format(image.shape))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        height, width = image.shape[:2]
        #print("input shape:{}".format(image.shape))
        # h,w,c -> c,h,w
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        ratio_w = new_w/w_origin
        ratio_h = new_h/h_origin
        return inputs, ratio_h, ratio_w

    def inference(self, image_path):
        # get model
        self.model = self._get_model(self.ckpt_path)
        # get input
        image = cv2.imread(image_path)
        inputs, ratio_h, ratio_w = self._preprocess(image) 
        ### inference  
        # origin output
        predictions = self.model([inputs])[0]["panoptic_seg"]
        # custom output
        #predictions = self.model([inputs])
        #print(predictions)
        # 防止gpu显存逐步累加
        torch.cuda.empty_cache()
        
        return predictions, ratio_h, ratio_w

def parse_panoptic_results(predictions, ratio_h, ratio_w):
    """
    (tensor([[5, 5, 5,  ..., 5, 5, 5],
        [5, 5, 5,  ..., 5, 5, 5],
        [5, 5, 5,  ..., 5, 5, 5],
        ...,
        [5, 5, 5,  ..., 5, 5, 5],
        [5, 5, 5,  ..., 5, 5, 5],
        [5, 5, 5,  ..., 5, 5, 5]], dtype=torch.int32), 
    [{'id': 1, 'isthing': True, 'score': 1.0, 'category_id': 0, 'instance_id': 0}, {'id': 2, 'isthing': True, 'score': 1.0, 'category_id': 0, 'instance_id': 1}, {'id': 3, 'isthing': True, 'score': 1.0, 'category_id': 0, 'instance_id': 2}, {'id': 4, 'isthing': True, 'score': 0.9999998807907104, 'category_id': 0, 'instance_id': 3}, {'id': 5, 'isthing': False, 'category_id': 1, 'area': 320506}])
    """
    print(predictions)
    panoptic_seg, segments_info  = predictions
    panoptic_seg = panoptic_seg.to("cpu")
    ### process _sinfo
    _seg = panoptic_seg
    _sinfo = {s["id"]: s for s in segments_info}  # seg id -> seg info
    #print(_sinfo)
    segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
    segment_ids = segment_ids.numpy()
    #print(segment_ids, areas)
    areas = areas.numpy()
    sorted_idxs = np.argsort(-areas)
    _seg_ids, _seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
    s_seg_ids = _seg_ids.tolist()
    #print(s_seg_ids)
    for sid, area in zip(_seg_ids, _seg_areas):
        if sid in _sinfo:
            _sinfo[sid]["area"] = float(area)
    #print(_sinfo)
    ### process instance masks
    def instance_masks(_seg, _seg_ids, _sinfo):
        for sid in _seg_ids:
            sinfo = _sinfo.get(sid)
            if sinfo is None or not sinfo["isthing"]:
                continue
            mask = (_seg == sid).numpy().astype(np.bool)
            if mask.sum() > 0:
                yield mask, sinfo
    ### get all instances second
    all_instances = list(instance_masks(_seg, _seg_ids, _sinfo))
    if len(all_instances) == 0:
        return _sinfo
    masks, sinfo = list(zip(*all_instances))
    category_ids = [x["category_id"] for x in sinfo]
    def _create_text_labels(classes, scores, class_names):
        """
        Args:
            classes (list[int] or None):
            scores (list[float] or None):
            class_names (list[str] or None):
        Returns:
            list[str] or None
        """
        labels = None
        if classes is not None and class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.2f}".format(s) for s in scores]
            else:
                labels = ["{}:{:.2f}".format(l, s) for l, s in zip(labels, scores)]
                
        return labels
    try:
        scores = [x["score"] for x in sinfo]
    except KeyError:
        scores = None
    thing_classes = ['text']
    labels = _create_text_labels(category_ids, scores, thing_classes)

    def get_min_rect(image):
        image = image[:,:,np.newaxis]*255
        image = np.array(image,np.uint8)
        _, contours, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minAreaRect = cv2.minAreaRect(contours[0])
        rectCnt = cv2.boxPoints(minAreaRect)
        return rectCnt
    results = []
    for i in range(len(masks)):
        mask = masks[i]
        image = cv2.imread(image_path)
        #print(image.shape)
        h,w,c  = image.shape
        mask = np.array(mask,np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        rectCnt = get_min_rect(mask)
        bbox = []
        for item in rectCnt:
            bbox.append(item[0])
            bbox.append(item[1])
        results.append({"id":i, "bbox":bbox, "score":labels[i].split(":")[1]})

    return results

def parse_panoptic_results_custom(predictions, ratio_h, ratio_w):
    """
    predictions: [[mask_array, score],...,[mask_array, score]]
    """
    def get_min_rect(image):
        image = image[:,:,np.newaxis]*255
        image = np.array(image,np.uint8)
        _, contours, hier = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minAreaRect = cv2.minAreaRect(contours[0])
        rectCnt = cv2.boxPoints(minAreaRect)
        return rectCnt
    results = []
    for i in range(len(predictions)):
        mask = predictions[i][0]
        score = predictions[i][1]
        mask = np.array(mask,np.uint8)
        #print(mask.shape)
        h,w = mask.shape
        origin_h = int(h/ratio_h) ; origin_w = int(w/ratio_w)
        mask = cv2.resize(mask, (origin_w, origin_h), interpolation=cv2.INTER_NEAREST)
        rectCnt = get_min_rect(mask)
        bbox = []
        for item in rectCnt:
            bbox.append(item[0])
            bbox.append(item[1])
        results.append([bbox, score])
    
    return results


def visualize_results(image_path, results, save_path,txt_save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image = cv2.imread(image_path)
    base_name = os.path.split(image_path)[1]
    prefix = base_name.split(".jpeg")[0]
    txt_save_path = os.path.join(txt_save_path, prefix+".txt")
    f = open(txt_save_path, "w")
    # 绘制检测结果
    im_show_1 = image.copy()
    for idx in range(len(results)):
        bbox = results[idx]["bbox"]
        score = results[idx]["score"]
        # 绘制矩形框
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        x3 = bbox[4]
        y3 = bbox[5]
        x4 = bbox[6]
        y4 = bbox[7]
        cv2.line(im_show_1, (x1, y1), (x2, y2), (255, 0, 0))
        cv2.line(im_show_1, (x2, y2), (x3, y3), (0, 255, 0))
        cv2.line(im_show_1, (x3, y3), (x4, y4), (0, 0, 255))
        cv2.line(im_show_1, (x4, y4), (x1, y1), (255, 255, 0))
        # 绘制text及logits
        start_x = int((x1 + x3)/2)
        start_y = int((y1 + y3)/2)
        str_line = score
        f.write(str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(x3)+","+str(y3)+","+str(x4)+","+str(y4)+"\n")
        cv2.putText(im_show_1,str_line,(start_x,start_y),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    f.close()
    base_name = os.path.split(image_path)[1]
    prefix = base_name.split(".")[0]
    image_save_path = os.path.join(save_path, prefix + "_det" + ".jpg")
    #cv2.imwrite(image_save_path, im_show_1)


if __name__ == "__main__":
    # define model path
    model_path = "/home/dell/zhanglimin/code/panoptic_seg/detectron2/final_model.pkl"
    # instantiate PnopticFPN
    p = PnopticFPN(model_path)
    # define image list
    #image_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20200723/coco/val2017"
    image_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/BENCHMARK/文本检测&OCR/OCR-20200102_en_仅框_labelme/"
    #image_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/BENCHMARK/cn-text-detection"
    #image_list = glob.glob(image_root+"/*/*/*/*.jpeg")
    #image_list = glob.glob("/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/temp22/OCR错误数据/ocr_ch_bad_cases_stg/*.jpg")
    #image_list = image_list*20
    print(image_list)
    # define save path
    save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/temp22/OCR错误数据/ocr_ch_bad_cases_stg_panoptic_seg_20200730"
    txt_ave_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/temp22/OCR错误数据/ocr_ch_bad_cases_stg_panoptic_seg_20200730_txt"
    for image_path in image_list:
        # inference
        predictions, ratio_h, ratio_w = p.inference(image_path)
        ### parse panoptic segmentation results
        # origin output
        results = parse_panoptic_results(predictions, ratio_h, ratio_w)
        # custom output
        #results = parse_panoptic_results_custom(predictions, ratio_h, ratio_w)
        #print(results)
        # visual results
        visualize_results(image_path, results, save_path, txt_ave_path)



