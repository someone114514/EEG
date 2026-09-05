# band.py  —— five-class band rejection for imagined speech
import torch
import torchaudio
import random
from typing import Tuple, List

class BandPretext(torch.nn.Module):
    """
    Band-stop pretext head (5 classes):
      LF:              0.5-8 Hz
      alpha_beta:      8-30 Hz
      gamma_low:       30-70 Hz
      gamma_high_low:  70-100 Hz
      gamma_high_high: 100-150 Hz  (ensure fs >= 300 Hz; better >= 500 Hz)
    """
    def __init__(self, input_dim: int, jitter_pct: float = 0.10):
        super().__init__()
        self.input_dim = input_dim
        self.jitter_pct = float(jitter_pct)

        # (low, high) in Hz — canonical edges before jitter
        self.bands: List[Tuple[float, float]] = [
            (3.0, 7.0),    # Theta (narrow)
            (8.0, 13.0),   # Mu
            (13.0, 30.0),  # Low Beta
            (30.0, 45.0),  # Low Gamma (optional but included here)
        ]

        self.classifier = torch.nn.Linear(
            in_features=input_dim,
            out_features=len(self.bands),
        )

    def _maybe_tensor_to_float(self, sfreq):
        # Accept scalar, 0-dim tensor, or shape [1] tensor (to match your code)
        if isinstance(sfreq, torch.Tensor):
            return float(sfreq.reshape(-1)[0].item())
        return float(sfreq)

    def _jitter_band(self, low: float, high: float) -> Tuple[float, float]:
        """Apply ±jitter_pct jitter while keeping low<high."""
        if self.jitter_pct <= 0:
            return low, high
        w = high - low
        dl = random.uniform(-self.jitter_pct, self.jitter_pct) * w
        dh = random.uniform(-self.jitter_pct, self.jitter_pct) * w
        new_low  = max(0.0, low  + dl)
        new_high = max(new_low + 0.1, high + dh)  # keep at least 0.1 Hz width
        return new_low, new_high

    def reject_band(self, x: torch.Tensor, sfreq) -> Tuple[torch.Tensor, int]:
        """
        Randomly reject one band on the *time* dimension of x using a biquad
        band-reject filter.  Returns (filtered_x, rejected_band_idx).

        Args:
            x:     shape [..., time]  (same as torchaudio expects)
            sfreq: sampling rate (float or Tensor)
        """
        fs = self._maybe_tensor_to_float(sfreq)
        idx = random.randrange(len(self.bands))
        low, high = self.bands[idx]
        low, high = self._jitter_band(low, high)

        # central freq & Q for torchaudio.functional.bandreject_biquad
        central = (low + high) / 2.0
        bandwidth = max(1e-6, high - low)
        Q = central / bandwidth

        # Safety: keep within Nyquist; if not, slightly pull back
        nyq = fs / 2.0
        if central >= nyq:
            # pull the band inside Nyquist with a small margin
            scale = (0.98 * nyq) / central
            central *= scale
            bandwidth *= scale
            Q = central / bandwidth

        x_out = torchaudio.functional.bandreject_biquad(
            x, sample_rate=fs, central_freq=central, Q=Q
        )
        return x_out, idx

    def forward(self, filtered_x: torch.Tensor, rejected_band_idx: int):
        """
        Keep the same API as your original code: pass filtered features to a
        linear classifier. If your encoder sits before this head, call it and
        pass its output as `filtered_x`.
        """
        z = self.classifier(filtered_x)

        label = torch.full(
            size=(filtered_x.shape[0],),
            fill_value=rejected_band_idx,
            dtype=torch.long,
            device=filtered_x.device,
        )
        loss = torch.nn.functional.cross_entropy(z, label)
        probs = torch.nn.functional.softmax(z, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return loss, preds, label
