import os
import SimpleITK as sitk
from radiomics import featureextractor

def extract_radiomic_features(image_path, mask_path, params_file=None):
    """
    Extracts radiomic features from a segmented ROI using pyradiomics.
    Requires image and mask as paths or SimpleITK objects.
    """
    # Initialize extractor
    if params_file and os.path.exists(params_file):
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    else:
        # Default settings
        extractor = featureextractor.RadiomicsFeatureExtractor()
        # Enable some feature classes
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()
        
    print(f"Extracting features from {image_path}...")
    
    # Run extraction
    result = extractor.execute(image_path, mask_path)
    
    # Process results: filter out diagnostic info
    features = {key: value for key, value in result.items() if not key.startswith('diagnostics')}
    
    return features

def save_features_to_csv(features, output_csv):
    import pandas as pd
    df = pd.DataFrame([features])
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
