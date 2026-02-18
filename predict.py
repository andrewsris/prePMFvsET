import slideflow as sf
import argparse
import os
import torch
import pandas as pd
from slideflow.mil import mil_config

def run_inference(feature_dir, model_dir, output_file, annotations=None, device='cuda'):
    """
    Performs MIL inference on pre-computed feature bags.

    Args:
        feature_dir (str): Path to the directory containing .pt feature bags.
        model_dir (str): Path to the Slideflow MIL model directory.
        output_file (str): Path to the CSV file where predictions will be saved.
        annotations (str, optional): Path to a user-provided CSV with 'bmbx_diagnosis' column. Required only for calculating performance metrics (e.g., AUC).
        device (str): Computing device ('cuda' or 'cpu').
    """
    
    # Initialize workspace
    project_root = os.path.join(feature_dir, 'inference_temp')
    if not os.path.exists(project_root):
        os.makedirs(project_root)
    
    P = sf.Project(project_root, name='Inference')

    # Identify available feature bags
    bag_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
    slide_names = [os.path.splitext(f)[0] for f in bag_files]
    
    if not slide_names:
        print(f"Error: No .pt feature files found in {feature_dir}.")
        return

    print(f"Detected {len(slide_names)} feature bags.")
    
    # Handle annotations
    if annotations:
        print(f"Using provided annotations for evaluation: {annotations}")
        annotations_file = annotations
    else:
        # Generate minimal annotation manifest if none provided
        annotations_file = os.path.join(project_root, 'annotations.csv')
        df = pd.DataFrame({'slide': slide_names})
        # If the user wants to calculate metrics, Slideflow requires bmbx_diagnosis.
        # We omit it here if not providing annotations.
        df.to_csv(annotations_file, index=False)
    
    # Configure dataset source
    P.add_source('Validation', slides=feature_dir)
    dataset = P.dataset(tile_px=299, tile_um=302, sources=['Validation'], annotations=annotations_file)
    dataset = dataset.filter({'slide': slide_names})
    
    print("Executing model inference...")
    
    # Perform evaluation
    # Note: outcomes='bmbx_diagnosis' will compute metrics if the column exists in the provided annotations
    results = P.evaluate_mil(
        model_dir,
        dataset=dataset,
        bags=feature_dir,
        outcomes='bmbx_diagnosis',
        save_predictions=True
    )
    
    if output_file:
        results.to_csv(output_file)
        print(f"Predictions successfully saved to {output_file}")
    else:
        print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Script for prePMF vs ET MIL Classifier')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing .pt feature bags')
    parser.add_argument('--model_dir', type=str, default='prePMFvsET_Classifier', help='Model directory path')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='Output CSV filename')
    parser.add_argument('--annotations', type=str, default=None, help='(Optional) CSV with bmbx_diagnosis for metrics calculation')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Computation device')
    
    args = parser.parse_args()
    
    run_inference(args.feature_dir, args.model_dir, args.output_file, args.annotations, args.device)
