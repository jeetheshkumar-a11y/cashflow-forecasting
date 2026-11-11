"""Optional zero-shot foundation model: Amazon Chronos-Bolt.

Chronos is a pretrained time-series foundation model: it forecasts a series
it has never seen, without any training on our data.  That makes it a
useful "newer model" comparison.

Requirements (not installed by default, ~1 GB with PyTorch):
    pip install chronos-forecasting torch

The model weights are downloaded from Hugging Face on first use, so this
needs internet access.  Run it with
    python scripts/run_experiment.py --with-chronos

NOTE: this wrapper was written against the ``chronos-forecasting`` 1.x API
(``BaseChronosPipeline.predict_quantiles``).  It could not be executed in the
environment where the rest of the results were produced (no access to
Hugging Face there), so Chronos numbers are not in the committed results.
"""
from __future__ import annotations

import numpy as np

from .baselines import Model


class ChronosModel(Model):
    name = "Chronos-Bolt (zero-shot)"

    def __init__(self, model_id="amazon/chronos-bolt-small", context=365, batch=64):
        self.model_id, self.context, self.batch = model_id, context, batch
        self.pipe = None

    def _load(self):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.pipe = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu",
                                                        torch_dtype=torch.float32)

    def _forecast(self, series_list, H):
        out = []
        for i in range(0, len(series_list), self.batch):
            ctx = [self.torch.tensor(s, dtype=self.torch.float32) for s in series_list[i:i + self.batch]]
            _, mean = self.pipe.predict_quantiles(context=ctx, prediction_length=H, quantile_levels=[0.5])
            out.append(mean.numpy())
        return np.clip(np.concatenate(out), 0, None)

    def predict(self, W, S):
        if self.pipe is None:
            self._load()
        H = W.y_inc.shape[1]
        ctx_i = [S.inc[u][max(0, t - self.context):t] for u, t in zip(W.user, W.origin)]
        ctx_e = [S.exp[u][max(0, t - self.context):t] for u, t in zip(W.user, W.origin)]
        return self._forecast(ctx_i, H), self._forecast(ctx_e, H)
