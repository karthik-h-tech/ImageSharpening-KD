import sys
import os
import torch
from collections import OrderedDict

# -------------------------------
# Add Restormer to sys.path
# -------------------------------
restormer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Restormer'))
if restormer_root not in sys.path:
    sys.path.insert(0, restormer_root)

from basicsr.models.archs.restormer_arch import Restormer


# -------------------------------
# Teacher Model
# -------------------------------
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
        output = self.model(x)
        return output[0] if isinstance(output, (tuple, list)) else output


# -------------------------------
# Load pretrained weights
# -------------------------------
def load_teacher_model(device):
    """
    Initializes TeacherNet and loads pretrained weights for MOTION DEBLURRING.
    """
    model = TeacherNet().to(device)

    # Path to pretrained motion deblurring weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(
        script_dir, '..', 'Restormer', 'Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth'
    )
    weights_path = os.path.normpath(weights_path)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"❌ Motion deblurring weight file not found at: {weights_path}")

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('params') or checkpoint.get('state_dict') or checkpoint
    else:
        state_dict = checkpoint

    # Clean 'module.' prefixes if trained with DataParallel
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state_dict[new_key] = v

    load_result = model.load_state_dict(cleaned_state_dict, strict=False)

    if load_result.missing_keys:
        print(f"⚠️ Missing keys:\n{load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"⚠️ Unexpected keys:\n{load_result.unexpected_keys}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Motion Deblurring Teacher loaded. Total parameters: {total_params:,}")
    model.eval()
    return model
