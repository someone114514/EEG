import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod
from .band import BandPretext
from .amp import AmpPretext
# from .reverse import ReversePretext
# from .temporal import TemporalPretext

class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Sequential()
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes)
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64*3*200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64*3*200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(64*3*200, 3*200),
                nn.ELU(),
                # nn.Dropout(param.dropout),
                nn.Linear(3*200, 200),
                nn.ELU(),
                # nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

        self.band = BandPretext(input_dim=3*200)
        self.amp = AmpPretext(input_dim=3*200)
        # self.temporal = TemporalPretext(input_dim=3*200, num_chunks=2, p_shuffle=0.5)
        # self.reverse = ReversePretext(input_dim=3*200)

    def forward(self, *args, **kwargs):
        x = kwargs.get('input_ids', None)
        if x is None:
            x = args[0]
            # raise ValueError("Expected 'input_ids' in kwargs but got None.")
    
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out

