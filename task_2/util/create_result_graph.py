import json
import os
import re # Regular expressions for extracting iteration numbers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Optional, but helps structure data for seaborn

# --- Configuration ---
SUMMARY_JSON_PATH = "/home/aakash/Desktop/carton_training/all_models_evaluation_results/evaluation_summary.json"
PLOT_OUTPUT_DIR = "/home/aakash/Desktop/carton_training/all_models_evaluation_results/plots" # Where to save plots
FIGURE_SIZE = (12, 7) # Width, Height in inches

# --- Helper Function ---
def extract_iteration(filename):
    """Extracts iteration number from model filenames like model_0010000.pth"""
    match = re.search(r'model_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))
    # Handle model_final.pth - assign a large number to place it at the end
    if filename == 'model_final.pth':
        return float('inf') # Represent infinity, will be handled later
    return None # Cannot extract iteration

# --- Main Plotting Logic ---
if __name__ == "__main__":
    # 1. Create output directory
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    print(f"Plots will be saved in: {PLOT_OUTPUT_DIR}")

    # 2. Load the summary data
    try:
        with open(SUMMARY_JSON_PATH, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded data from: {SUMMARY_JSON_PATH}")
    except FileNotFoundError:
        print(f"Error: Summary file not found at {SUMMARY_JSON_PATH}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {SUMMARY_JSON_PATH}")
        exit(1)

    # 3. Parse and structure the data
    parsed_data = []
    max_iteration = 0
    found_final = False

    for model_name, results in all_results.items():
        iteration = extract_iteration(model_name)
        if iteration is None:
            print(f"Warning: Could not extract iteration from '{model_name}'. Skipping.")
            continue
        if iteration != float('inf'):
             max_iteration = max(max_iteration, iteration)
        else:
             found_final = True

        # Extract bbox metrics if available
        if 'bbox' in results and isinstance(results['bbox'], dict):
            for metric, value in results['bbox'].items():
                parsed_data.append({
                    'model': model_name,
                    'iteration': iteration,
                    'task': 'bbox',
                    'metric': metric,
                    'value': value
                })

        # Extract segm metrics if available
        if 'segm' in results and isinstance(results['segm'], dict):
             for metric, value in results['segm'].items():
                parsed_data.append({
                    'model': model_name,
                    'iteration': iteration,
                    'task': 'segm',
                    'metric': metric,
                    'value': value
                })

    if not parsed_data:
        print("Error: No valid data found to plot after parsing.")
        exit(1)

    # Replace inf for 'model_final.pth' with max_iteration + step (e.g., +1000 or just +1)
    # This ensures it appears last on the plot. Calculate a reasonable step if possible.
    step = 1000 # Default step
    iterations_present = sorted([d['iteration'] for d in parsed_data if d['iteration'] != float('inf')])
    if len(iterations_present) > 1:
        step = iterations_present[1] - iterations_present[0] # Estimate step from data

    final_iteration_value = max_iteration + step if found_final else max_iteration

    for d in parsed_data:
        if d['iteration'] == float('inf'):
            d['iteration'] = final_iteration_value

    # Convert to Pandas DataFrame for easier plotting with Seaborn
    df = pd.DataFrame(parsed_data)

    # Sort by iteration for correct line plotting
    df = df.sort_values(by='iteration')

    # 4. Create Plots

    # --- Bounding Box Plot ---
    plt.figure(figsize=FIGURE_SIZE)
    bbox_df = df[df['task'] == 'bbox']
    # Plot major AP metrics
    sns.lineplot(data=bbox_df[bbox_df['metric'].isin(['AP', 'AP50', 'AP75'])],
                 x='iteration', y='value', hue='metric', marker='o')
    plt.title('Bounding Box AP Metrics vs. Training Iteration')
    plt.xlabel('Training Iteration')
    plt.ylabel('AP Score')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Metric')
    plt.tight_layout()
    for line in bbox_df[bbox_df['metric'].isin(['AP', 'AP50', 'AP75'])].itertuples():
        plt.text(line.iteration, line.value, f"{line.value:.2f}", fontsize=8, ha='right')
    bbox_plot_path = os.path.join(PLOT_OUTPUT_DIR, 'bbox_metrics_vs_iteration.png')
    plt.savefig(bbox_plot_path)
    print(f"Saved BBox plot to: {bbox_plot_path}")
    # plt.show() # Uncomment to display plot interactively

    # --- Segmentation Plot ---
    plt.figure(figsize=FIGURE_SIZE)
    segm_df = df[df['task'] == 'segm']
    if not segm_df.empty:
        # Plot major AP metrics
        sns.lineplot(data=segm_df[segm_df['metric'].isin(['AP', 'AP50', 'AP75'])],
                     x='iteration', y='value', hue='metric', marker='o')
        plt.title('Segmentation AP Metrics vs. Training Iteration')
        plt.xlabel('Training Iteration')
        plt.ylabel('AP Score')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Metric')
        plt.tight_layout()
        for line in segm_df[segm_df['metric'].isin(['AP', 'AP50', 'AP75'])].itertuples():
            plt.text(line.iteration, line.value, f"{line.value:.2f}", fontsize=8, ha='right')
        segm_plot_path = os.path.join(PLOT_OUTPUT_DIR, 'segm_metrics_vs_iteration.png')
        plt.savefig(segm_plot_path)
        print(f"Saved Segm plot to: {segm_plot_path}")
        # plt.show() # Uncomment to display plot interactively
    else:
        print("Skipping segmentation plot as no segmentation data was found.")

    # --- Optional: Plot Small, Medium, Large AP ---
    plt.figure(figsize=FIGURE_SIZE)
    bbox_size_df = df[df['task'] == 'bbox']
    if not bbox_size_df.empty and all(m in bbox_size_df['metric'].unique() for m in ['APs', 'APm', 'APl']):
         sns.lineplot(data=bbox_size_df[bbox_size_df['metric'].isin(['APs', 'APm', 'APl'])],
                     x='iteration', y='value', hue='metric', marker='o')
         plt.title('Bounding Box Size AP Metrics (Small, Medium, Large) vs. Training Iteration')
         plt.xlabel('Training Iteration')
         plt.ylabel('AP Score')
         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
         plt.legend(title='Metric (Size)')
         plt.tight_layout()
         for line in bbox_size_df[bbox_size_df['metric'].isin(['APs', 'APm', 'APl'])].itertuples():
            plt.text(line.iteration, line.value, f"{line.value:.2f}", fontsize=8, ha='right')
         bbox_size_plot_path = os.path.join(PLOT_OUTPUT_DIR, 'bbox_size_metrics_vs_iteration.png')
         plt.savefig(bbox_size_plot_path)
         print(f"Saved BBox Size plot to: {bbox_size_plot_path}")
         # plt.show()
    else:
         print("Skipping BBox size plot (APs, APm, APl not consistently found).")

    # Repeat for Segmentation Size AP if needed following the same pattern

    plt.close('all') # Close all figures
    print("\nPlotting complete.")
