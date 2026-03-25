import os
import pydicom
import numpy as np
import SimpleITK as sitk

def load_dbt_dicom(dicom_path):
    """
    Loads a DBT DICOM file and returns the pixel data as a 3D numpy array.
    Handles both multi-frame DICOMs and single-frame stacks.
    """
    ds = pydicom.dcmread(dicom_path)
    
    # Check if multi-frame (typical for DBT)
    if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
        pixel_array = ds.pixel_array
    else:
        # If it's a directory of single-slice DICOMs (less common for DBT, but possible)
        if os.path.isdir(dicom_path):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            pixel_array = sitk.GetArrayFromImage(image)
        else:
            pixel_array = ds.pixel_array[np.newaxis, ...]
            
    return pixel_array, ds

def save_as_nifti(pixel_array, output_path, reference_ds=None):
    """
    Saves a 3D numpy array as a NIfTI file for easier ML processing.
    """
    image = sitk.GetImageFromArray(pixel_array.astype(np.float32))
    # Optionally set spacing/origin from DICOM
    if reference_ds:
        try:
            spacing = (float(reference_ds.PixelSpacing[0]), 
                       float(reference_ds.PixelSpacing[1]), 
                       float(reference_ds.SliceThickness))
            image.SetSpacing(spacing)
        except AttributeError:
            pass
            
    sitk.WriteImage(image, output_path)
    print(f"Saved volume to {output_path}")
