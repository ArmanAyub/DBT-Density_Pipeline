import numpy as np
import cv2
from scipy.ndimage import median_filter
from skimage import exposure, filters

def normalize_intensity(volume, range_min=0, range_max=1):
    """
    Standardizes volume intensity to a given range [range_min, range_max].
    """
    v_min, v_max = volume.min(), volume.max()
    normalized = (volume - v_min) / (v_max - v_min + 1e-7)
    normalized = normalized * (range_max - range_min) + range_min
    return normalized

def denoise_volume(volume, size=3):
    """
    Applies a median filter for noise reduction.
    """
    return median_filter(volume, size=size)

def generate_breast_mask(volume):
    """
    Generates a binary mask to isolate breast tissue from the background.
    Uses Otsu's thresholding on a slice-by-slice or global basis.
    """
    mask = np.zeros_like(volume, dtype=np.uint8)
    for i in range(volume.shape[0]):
        slice_img = volume[i]
        # Skip empty slices
        if slice_img.max() == slice_img.min():
            continue
        
        # Apply Otsu's threshold
        thresh = filters.threshold_otsu(slice_img)
        binary = slice_img > thresh
        
        # Post-processing: remove small noise, fill holes
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # Keep the largest connected component as the breast
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask[i] = (labels == largest_label).astype(np.uint8)
            
    return mask

def adaptive_histogram_equalization(volume):
    """
    Enhances contrast using CLAHE.
    """
    enhanced = np.zeros_like(volume, dtype=np.float32)
    for i in range(volume.shape[0]):
        enhanced[i] = exposure.equalize_adapthist(volume[i])
    return enhanced
