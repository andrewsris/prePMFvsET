import slideflow as sf
from slideflow.slide import qc
import argparse
import os
import shutil

# HISTOPATHOLOGICAL STAIN NORMALIZATION PARAMETERS
# Standardized Macenko stain matrix and concentrations utilized during model training
STAIN_MATRIX_TARGET = [
    [0.5062568187713623, 0.22186939418315887],
    [0.7532230615615845, 0.8652154803276062],
    [0.4069173336029053, 0.42241501808166504]
]
TARGET_CONCENTRATIONS = [1.7656903266906738, 1.2797492742538452]

def get_preprocessing_project(root_dir):
    """
    Initializes a programmatic Slideflow project for automated preprocessing.

    Args:
        root_dir (str): Root directory for the temporary project workspace.
    
    Returns:
        slideflow.Project: Initialized Slideflow project object.
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    if os.path.exists(os.path.join(root_dir, 'settings.json')):
        P = sf.Project(root_dir)
    else:
        P = sf.Project(root_dir, name='prePMF_Preprocessing')
    
    return P

def run_preprocessing(wsi_dir, output_dir, device='cuda'):
    """
    Executes the histopathology image preprocessing pipeline.

    Workflow:
    1. Workspace initialization.
    2. Dataset source configuration.
    3. Tile extraction at 302 µm (299 px) resolution.
    4. Feature extraction using RetCCL with Macenko normalization.

    Args:
        wsi_dir (str): Directory containing Whole Slide Images (e.g., .svs, .ndpi).
        output_dir (str): Destination directory for extracted feature bags (.pt).
        device (str): Computation device for deep learning tasks ('cuda' or 'cpu').
    """
    
    # Initialize Project Workspace
    project_temp = os.path.join(output_dir, 'preprocessing_temp')
    P = get_preprocessing_project(project_temp)
    
    # Configure Dataset Source
    source_name = 'Evaluation_Source'
    P.add_source(
        source_name, 
        slides=wsi_dir, 
        tiles=os.path.join(project_temp, 'tiles'), 
        tfrecords=os.path.join(project_temp, 'tfrecords')
    )
    
    dataset = P.dataset(tile_px=299, tile_um=302, sources=[source_name])
    print(f"Dataset identified: {len(dataset.slide_paths())} slides found.")
    
    # Configure Normalization
    normalizer = sf.norm.autoselect("macenko", backend="torch")
    if device == 'cuda':
         normalizer.device = 'gpu'
    
    # Apply study-specific target parameters
    normalizer.stain_matrix_target = STAIN_MATRIX_TARGET
    normalizer.target_concentrations = TARGET_CONCENTRATIONS
    
    # Feature Extraction (RetCCL Workflow)
    print("Initiating feature extraction via RetCCL...")
    extractor = sf.model.build_feature_extractor('retccl', tile_px=299)
    
    feature_set = sf.DatasetFeatures(extractor, dataset=dataset, normalizer=normalizer)
    
    # Export generated features in PyTorch format
    feature_set.to_torch(output_dir)
    print(f"Preprocessing completed. Feature bags generated in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Histopathology Preprocessing Pipeline for prePMF vs ET MIL Classifier')
    parser.add_argument('--wsi_dir', type=str, required=True, help='Input directory containing Whole Slide Images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for feature bag storage')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Computation device selection')
    
    args = parser.parse_args()
    
    run_preprocessing(args.wsi_dir, args.output_dir, args.device)
