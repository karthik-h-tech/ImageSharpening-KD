import sys
import os
import torch
from collections import OrderedDict

# Add Restormer root folder to sys.path for correct import of basicsr
restormer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Restormer'))
if restormer_root not in sys.path:
    sys.path.insert(0, restormer_root)

from basicsr.models.archs.restormer_arch import Restormer


class TeacherNet(torch.nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.model = Restormer(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',
            dual_pixel_task=False
        )

    def forward(self, x):
        return self.model(x)


def load_teacher_model(device):
    """
    Initialize TeacherNet and load pretrained Restormer weights from
    'single_image_defocus_deblurring.pth' located in the same folder as this script.

    Args:
        device (torch.device): 'cuda' or 'cpu'

    Returns:
        model (torch.nn.Module): Loaded teacher model in eval mode.
    """
    model = TeacherNet().to(device)

    # Weights path is assumed to be alongside this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, 'single_image_defocus_deblurring.pth')

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weight file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # Extract state_dict from checkpoint variations
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (for DataParallel compatibility)
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v

    load_result = model.load_state_dict(cleaned_state_dict, strict=False)

    if load_result.missing_keys:
        print(f"⚠️ Missing keys when loading weights:\n{load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"⚠️ Unexpected keys when loading weights:\n{load_result.unexpected_keys}")

    model.eval()
    print("✅ Teacher model loaded successfully.")
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = load_teacher_model(device)
