import os
import argparse
from src.preprocessing.dicom_loader import load_dbt_dicom, save_as_nifti
from src.preprocessing.preprocessing_utils import normalize_intensity, denoise_volume, generate_breast_mask
from src.density.density_estimator import estimate_vbd, generate_density_heatmap

def run_pipeline(dicom_path, output_dir):
    """
    Main pipeline to process a DBT DICOM file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loading
    print("Step 1: Loading DBT DICOM...")
    pixel_array, ds = load_dbt_dicom(dicom_path)
    
    # 2. Preprocessing
    print("Step 2: Preprocessing...")
    normalized = normalize_intensity(pixel_array)
    denoised = denoise_volume(normalized)
    breast_mask = generate_breast_mask(denoised)
    
    # 3. Volumetric Breast Density Estimation
    print("Step 3: Estimating VBD...")
    vbd, dense_mask, fgt_threshold = estimate_vbd(denoised, breast_mask)
    print(f"Estimated VBD: {vbd:.2%}")
    
    # 4. Visualization
    print("Step 4: Generating Density Heatmap...")
    heatmap_path = os.path.join(output_dir, "density_heatmap.png")
    generate_density_heatmap(denoised, breast_mask, output_path=heatmap_path)
    
    # 5. Save Results
    print("Step 5: Saving processed volume and masks...")
    save_as_nifti(denoised, os.path.join(output_dir, "volume.nii.gz"), reference_ds=ds)
    save_as_nifti(breast_mask, os.path.join(output_dir, "breast_mask.nii.gz"), reference_ds=ds)
    save_as_nifti(dense_mask.astype(float), os.path.join(output_dir, "dense_mask.nii.gz"), reference_ds=ds)
    
    print(f"Pipeline finished. Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBT Breast Analysis Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input DBT DICOM")
    parser.add_argument("--output", type=str, required=True, help="Directory to save results")
    args = parser.parse_args()
    
    run_pipeline(args.input, args.output)
