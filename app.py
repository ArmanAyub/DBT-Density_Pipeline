import streamlit as st
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from src.preprocessing.dicom_loader import load_dbt_dicom
from src.preprocessing.preprocessing_utils import normalize_intensity, generate_breast_mask
from src.density.density_estimator import estimate_vbd, generate_density_heatmap

st.set_page_config(page_title="DBT Breast Analysis Dashboard", layout="wide")

st.title(" Automated Breast Density & Mass Analysis")
st.markdown("""
This dashboard allows you to analyze 3D Digital Breast Tomosynthesis (DBT) volumes. 
Upload a DICOM or NIfTI file to calculate volumetric density and detect candidate masses.
""")

# Sidebar for settings
st.sidebar.header("Settings")
analysis_mode = st.sidebar.selectbox("Analysis Mode", ["Full Pipeline", "Density Only", "Mass Segmentation"])

# File Uploader
uploaded_file = st.file_uploader("Upload DBT DICOM or NIfTI", type=["dcm", "nii", "nii.gz"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = os.path.join("data", "temp_upload" + os.path.splitext(uploaded_file.name)[1])
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # Step 1: Loading
    with st.spinner("Processing volume..."):
        if uploaded_file.name.endswith((".nii", ".nii.gz")):
            image = sitk.ReadImage(temp_path)
            volume = sitk.GetArrayFromArray(image)
        else:
            volume, ds = load_dbt_dicom(temp_path)
        
        normalized = normalize_intensity(volume)
        breast_mask = generate_breast_mask(normalized)
        vbd, dense_mask, _ = estimate_vbd(normalized, breast_mask)

    # Layout for visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("3D Volume Viewer")
        slice_idx = st.slider("Select Slice", 0, volume.shape[0]-1, volume.shape[0]//2)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(normalized[slice_idx], cmap='gray')
        ax[0].set_title(f"Original Slice {slice_idx}")
        ax[0].axis('off')
        
        # Show mask overlay if available
        overlay = normalized[slice_idx].copy()
        mask_slice = dense_mask[slice_idx]
        ax[1].imshow(normalized[slice_idx], cmap='gray')
        ax[1].imshow(mask_slice, cmap='jet', alpha=0.3 if mask_slice.any() else 0)
        ax[1].set_title("Density Mask Overlay")
        ax[1].axis('off')
        
        st.pyplot(fig)

    with col2:
        st.subheader("Analysis Results")
        st.metric("Volumetric Breast Density (VBD)", f"{vbd:.2%}")
        
        # Density Heatmap
        st.write("Spatial Density Heatmap")
        heatmap = np.mean(normalized, axis=0)
        fig_h, ax_h = plt.subplots()
        im = ax_h.imshow(heatmap, cmap='hot')
        plt.colorbar(im, ax=ax_h)
        ax_h.axis('off')
        st.pyplot(fig_h)

    # Placeholder for segmentation
    if analysis_mode == "Mass Segmentation":
        st.info("Mass Segmentation Model is ready for training. Once trained, results will appear here.")

else:
    st.info("Please upload a DBT volume to begin the analysis.")

# Cleanup temp file
# if os.path.exists(temp_path):
#     os.remove(temp_path)
