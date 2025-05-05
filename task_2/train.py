import os
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

# Register the dataset
train_annotation_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/annotations/instances_train2017.json"
train_image_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/images/train2017"
val_image_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/images/val2017"
val_annotation_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/annotations/instances_val2017.json"
register_coco_instances("trainDataset", {}, train_annotation_path, train_image_path)
register_coco_instances("valDataset", {}, val_annotation_path, val_image_path)

# Define the model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("trainDataset",)
cfg.DATASETS.TEST = ("valDataset",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
cfg.INPUT.FLIP = True
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.DATALOADER.NUM_WORKERS = 4
cfg.DATALOADER.SHUFFLE = True
num_images = 7400 
batch_size = cfg.SOLVER.IMS_PER_BATCH
iterations_per_epoch = num_images // batch_size  
epochs = 20  
cfg.SOLVER.MAX_ITER = iterations_per_epoch * epochs  
cfg.SOLVER.BASE_LR = 0.0005  
cfg.SOLVER.STEPS = [
    int(cfg.SOLVER.MAX_ITER * 0.3),
    int(cfg.SOLVER.MAX_ITER * 0.7)
]  
cfg.SOLVER.GAMMA = 0.1 
cfg.SOLVER.CHECKPOINT_PERIOD = iterations_per_epoch 
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.TEST.EVAL_PERIOD = 1000
cfg.MODEL.DEVICE = "cuda"
cfg.OUTPUT_DIR = "/home/aakash/Desktop/carton_training/trained_model"
cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION = "/home/aakash/Desktop/carton_training/trained_model/validation_set_evaluation"

class CocoTrainer(DefaultTrainer):
    """
    A custom trainer class that evaluates the model on the validation set every `_C.TEST.EVAL_PERIOD` iterations.
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION,
                        exist_ok=True)

        return COCOEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION)

setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "training-log.txt"))

if __name__ == "__main__":
    # Save the configuration to a YAML file
    output_yaml_path = os.path.join(
        "/home/aakash/Desktop/carton_training/trained_model",
        "train_config.yaml",
    )

    
    with open(output_yaml_path, "w") as f:
        f.write(cfg.dump())
        print(f"Configuration saved to {output_yaml_path}")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
