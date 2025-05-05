from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
import os

setup_logger()

def get_config():
    cfg = get_cfg()
    cfg.merge_from_file("/home/aakash/Desktop/carton_training/trained_model/train_config.yaml")
    cfg.MODEL.WEIGHTS = "/home/aakash/Desktop/carton_training/trained_model/model_final.pth"
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def reg():
    val_image_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/images/val2017"
    val_annotation_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/annotations/instances_val2017.json"
    register_coco_instances("valDataset", {}, val_annotation_path, val_image_path)
    
def draw_pred_on_images(cfg):
    file_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/images/val2017"
    metadata = MetadataCatalog.get("valDataset")
    predictor = DefaultPredictor(cfg)
    for i, file in enumerate(os.listdir(file_path)):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image = cv2.imread(os.path.join(file_path, file))
            outputs = predictor(image)
            v = Visualizer(
                image[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.SEGMENTATION,
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            outputImage = out.get_image()[:, :, ::-1]
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("output", 800, 600)
            cv2.imshow("output", outputImage)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cfg = get_config()
    reg()
    draw_pred_on_images(cfg)
