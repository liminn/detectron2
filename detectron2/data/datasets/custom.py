from .register_coco import register_coco_panoptic_separated
"""
register_coco_panoptic_separated(name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json)
"""
name = "custom_card_train"
metadata = {}
image_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/coco/train2017"
panoptic_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/panoptic_train2017" 
panoptic_json = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/annotations/panoptic_train2017.json"
sem_seg_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/panoptic_stuff_train2017"
instances_json = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/coco/annotations/instances_train2017.json"
register_coco_panoptic_separated(name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json)

name = "custom_card_val"
metadata = {}
image_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/coco/val2017"
panoptic_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/panoptic_val2017" 
panoptic_json = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/annotations/panoptic_val2017.json"
sem_seg_root = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/panoptic/panoptic_stuff_val2017"
instances_json = "/media/dell/6e8a7942-5a27-4e56-bffe-1af5a12aabb4/data/card_panoptic_20200718/coco/annotations/instances_val2017.json"
register_coco_panoptic_separated(name, metadata, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json)

