from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import glob
import cv2
import os
import numpy as np

if __name__ == "__main__":
    
    # Inference with a panoptic segmentation model
    image_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/text_panoptic_20201113/coco/val2017"
    image_list = glob.glob(image_root+"/*.jpg")
    print(image_list)
    save_path = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/train_results/panoptic_seg/detectron2/text_20210320_val/inference_visible"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cfg = get_cfg()
    ### card
    #cfg.merge_from_file("../configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_card.yaml")
    #cfg.MODEL.WEIGHTS = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/train_results/panoptic_seg/detectron2/card_20200722/model_final.pth"
    ### text
    cfg.merge_from_file("/home/dell/zhanglimin/code/panoptic_seg/detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x_text.yaml")
    #cfg.MODEL.WEIGHTS = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/train_results/panoptic_seg/detectron2/text_res50_changed_min256_max_1333_20201203/model_final.pth"
    cfg.MODEL.WEIGHTS = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/train_results/panoptic_seg/detectron2/text_20201113/model_final.pth"
    predictor = DefaultPredictor(cfg)
    for image_path in image_list:
        image = cv2.imread(image_path)
        base_name = os.path.split(image_path)[1]
        #pre_results = predictor(image) 
        #print(pre_results.keys()) # dict_keys(['sem_seg', 'instances', 'panoptic_seg'])
        panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
        #print(panoptic_seg.shape) # (height, width)
        #print(segments_info)      # instance info
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        image_save_path = os.path.join(save_path, base_name.split(".jpg")[0]+"_src.jpg")
        res_save_path = os.path.join(save_path, base_name)
        #cv2_imshow(out.get_image()[:, :, ::-1])
        print("save {}".format(image_save_path))
        #cv2.imwrite(res_save_path, out.get_image()[:, :, ::-1])
        image_out = out.get_image()[:, :, ::-1]
        image_resize = cv2.resize(image, (image_out.shape[1], image_out.shape[0]))
        image_stack = np.hstack([image_resize, image_out])
        cv2.imwrite(image_save_path, image_stack)
        