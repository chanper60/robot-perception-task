import detectron2
from detectron2.utils.logger import setup_logger
# setup_logger() # Don't set up global logger here, do it per model

# import some common libraries
import numpy as np
import os, json, cv2, random, glob, logging

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

# --- Configuration Section ---

# Paths
BASE_MODEL_DIR = "/home/aakash/Desktop/carton_training/trained_model" # Directory containing all .pth files
BASE_CONFIG_FILE = os.path.join(BASE_MODEL_DIR, "train_config.yaml") # Assumes config is in the same dir
MAIN_OUTPUT_DIR = "/home/aakash/Desktop/carton_training/all_models_evaluation_results" # Main dir for all outputs
DATASET_NAME = "valDataset" # Use a descriptive name

# Evaluation Settings
SCORE_THRESH_TEST = 0.5
DEVICE = "cuda" 

setup_logger(output="/home/aakash/Desktop/carton_training/all_models_evaluation_results/evaluation-log.txt")


def get_base_config(config_file):
    """Loads the base configuration from the training YAML file."""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = DEVICE
    return cfg

def register_validation_dataset():
    """Registers the COCO validation dataset if not already registered."""
    val_image_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/images/val2017"
    val_annotation_path = "/home/aakash/Desktop/OSCD/coco_carton/oneclass_carton/annotations/instances_val2017.json"
    register_coco_instances("valDataset", {}, val_annotation_path, val_image_path)

def evaluate_single_model(base_cfg, model_path, dataset_name, score_thresh, output_dir):
    """Evaluates a single model checkpoint."""
    model_filename = os.path.basename(model_path)
    model_eval_output_dir = os.path.join(output_dir, f"{model_filename}_eval")
    os.makedirs(model_eval_output_dir, exist_ok=True)

    # Configure logging for this specific model evaluation
    log_file = os.path.join(model_eval_output_dir, "val-log.txt")
    print(f"--- Evaluating Model: {model_filename} ---")
    print(f"Output Directory: {model_eval_output_dir}")
    print(f"Log File: {log_file}")

    # Create a specific config for this model
    cfg = base_cfg.clone() # Important: clone the base config
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.OUTPUT_DIR = model_eval_output_dir # Set output dir for any potential internal saves

    print(f"Using Model Weights: {cfg.MODEL.WEIGHTS}")
    print(f"Using Score Threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    print(f"Using Device: {cfg.MODEL.DEVICE}")

    try:
        # Build predictor and evaluator
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(dataset_name, output_dir=model_eval_output_dir)
        val_loader = build_detection_test_loader(cfg, dataset_name)

        # Run inference
        print(f"Starting inference for {model_filename} on dataset {dataset_name}...")
        results = inference_on_dataset(predictor.model, val_loader, evaluator)
        print(f"Finished inference for {model_filename}.")
        print(f"Evaluation results: {results}")

        # Ensure logs specific to this model are flushed if needed (usually automatic on exit)

        return results

    except Exception as e:
        print(f"Error evaluating model {model_filename}: {e}", exc_info=True)
        return {"error": str(e)} # Return an error indicator

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Create main output directory
    os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)
    print(f"Main output directory: {MAIN_OUTPUT_DIR}")

    # 2. Register the validation dataset (only needs to be done once)
    register_validation_dataset()

    # 3. Load base configuration
    print(f"Loading base configuration from: {BASE_CONFIG_FILE}")
    base_cfg = get_base_config(BASE_CONFIG_FILE)


    # 4. Find model files
    model_files = sorted(glob.glob(os.path.join(BASE_MODEL_DIR, "model_*.pth")))
    # remove model_final.pth from the list if it exists
    model_files = [mf for mf in model_files if "model_final" not in mf]
    if not model_files:
        print(f"Error: No model files matching 'model_*.pth' found in {BASE_MODEL_DIR}")
        exit(1)

    print(f"Found {len(model_files)} model checkpoints to evaluate:")
    for mf in model_files:
        print(f" - {os.path.basename(mf)}")

    # 5. Evaluate each model and store results
    all_evaluation_results = {}
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\n--- Processing: {model_name} ---")

        # Evaluate the model
        eval_results = evaluate_single_model(
            base_cfg=base_cfg,
            model_path=model_path,
            dataset_name=DATASET_NAME,
            score_thresh=SCORE_THRESH_TEST,
            output_dir=MAIN_OUTPUT_DIR # Pass the *main* output dir here
        )

        # Store results
        all_evaluation_results[model_name] = eval_results
        print(f"--- Finished: {model_name} ---")


    # 6. Save aggregated results to JSON
    summary_file_path = os.path.join(MAIN_OUTPUT_DIR, "evaluation_summary.json")
    print(f"\nSaving aggregated evaluation results to: {summary_file_path}")
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(all_evaluation_results, f, indent=4)
        print("Successfully saved evaluation summary.")
    except Exception as e:
        print(f"Error saving summary file: {e}")

    print("\nEvaluation complete.")
