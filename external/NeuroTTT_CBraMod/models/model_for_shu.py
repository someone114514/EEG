import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .cbramod import CBraMod
from .band_for_mi import BandPretext
from .amp import AmpPretext
from .phase import PhasePretext
from .temporal import TemporalPretext
from .filp_for_shu import JigsawPretext


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
        self.backbone.proj_out = nn.Identity()
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 4 * 200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(32 * 4 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, 1),
                Rearrange('b 1 -> (b 1)'),
            )

        self.band = BandPretext(input_dim=4*200)
        self.amp = AmpPretext(input_dim=4*200)
        self.phase = PhasePretext(input_dim=4*200)
        self.temporal = TemporalPretext(input_dim=4*200, num_chunks=2, p_shuffle=0.5)
        self.jigsaw = JigsawPretext()

        # if param.use_finetuned_weights:
        #     map_location = torch.device(f'cuda:{param.cuda}')
        #     print(f"Loading finetuned weights from {param.foundation_dir}")
        #     self.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))


    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


