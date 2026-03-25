import numpy as np
import matplotlib.pyplot as plt
from skimage import filters

def estimate_vbd(volume, mask):
    """
    Computes Volumetric Breast Density (VBD).
    VBD = (Volume of Fibroglandular Tissue) / (Total Breast Volume)
    """
    # Isolate breast tissue
    breast_tissue = volume[mask > 0]
    
    # Estimate threshold for dense tissue (FGT)
    # This could be more advanced (e.g. adaptive thresholding)
    try:
        fgt_threshold = filters.threshold_otsu(breast_tissue)
    except ValueError:
        fgt_threshold = 0.5 # fallback if volume is empty/corrupt
        
    dense_mask = (volume > fgt_threshold) & (mask > 0)
    
    total_voxels = np.sum(mask)
    dense_voxels = np.sum(dense_mask)
    
    vbd = dense_voxels / total_voxels if total_voxels > 0 else 0
    return vbd, dense_mask, fgt_threshold

def generate_density_heatmap(volume, mask, output_path=None):
    """
    Creates a 2D density projection/heatmap of the breast.
    Aggregates dense tissue across slices for spatial visualization.
    """
    # Project the intensity values across slices
    # Could also use Maximum Intensity Projection (MIP)
    heatmap = np.mean(volume, axis=0)
    
    plt.figure(figsize=(8,8))
    plt.imshow(heatmap, cmap='hot')
    plt.title('Breast Density Heatmap')
    plt.colorbar(label='Mean Intensity')
    plt.axis('off')
    if output_path:
        plt.savefig(output_path)
    plt.close()
    
    return heatmap

def get_candidate_dense_regions(dense_mask, min_size=100, min_persistence=3):
    """
    Detects candidate dense regions (potential masses) using CCA and spatial persistence.
    """
    import scipy.ndimage as nd
    
    # Perform 3D Connected Component Analysis
    labeled_array, num_features = nd.label(dense_mask)
    
    candidate_regions = []
    
    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        size = np.sum(region)
        
        # Check size constraint
        if size < min_size:
            continue
            
        # Check persistence (how many slices it spans)
        slices = np.any(region, axis=(1, 2))
        persistence = np.sum(slices)
        
        if persistence < min_persistence:
            continue
            
        candidate_regions.append(region)
        
    return candidate_regions
