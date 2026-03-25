import torch
import torch.nn as nn
from monai.networks.nets import UNet

class UNet25D(nn.Module):
    """
    A 2.5D U-Net model for DBT segmentation.
    Expects N-channel input where channels correspond to adjacent slices.
    """
    def __init__(self, in_channels=3, out_channels=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)):
        super(UNet25D, self).__init__()
        
        self.model = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
            dropout=0.2
        )
        
    def forward(self, x):
        """
        Input x: [Batch, Channels, Height, Width]
        Returns segmented mask for the central slice.
        """
        return self.model(x)

def load_model(weights_path=None, device='cpu'):
    model = UNet25D()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    return model

def create_training_dataset(volume, labels, slice_radius=1):
    """
    Prepares 2.5D patches from a 3D volume.
    Each patch is (2*slice_radius + 1) channels.
    """
    # ... logic for patch extraction ...
    pass
