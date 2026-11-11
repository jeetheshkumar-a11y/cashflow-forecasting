"""Deep models in TensorFlow / Keras: LSTM, GRU and N-BEATS.

All three are *global* models: one network is trained on windows from all
users (each window scaled by its own recent average flow) and directly
outputs the next H days of income and expense (a 2H vector).
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from ..features import Series, Windows
from .baselines import Model


def _keras():
    import keras
    return keras


def build_rnn(L, F, H, cell="lstm", units=64, dropout=0.1):
    keras = _keras()
    layers = keras.layers
    Cell = layers.LSTM if cell == "lstm" else layers.GRU
    inp = keras.Input((L, F))
    x = Cell(units, return_sequences=True, dropout=dropout)(inp)
    x = Cell(units // 2, dropout=dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(2 * H, activation="softplus")(x)
    return keras.Model(inp, out, name=cell)


def build_nbeats(L, F, H, n_blocks=6, width=256, n_layers=4):
    """Generic N-BEATS (Oreshkin et al., 2020): a stack of fully connected
    blocks with doubly-residual connections.  Each block removes the part of
    the input it can explain (backcast) and adds its forecast."""
    keras = _keras()
    layers = keras.layers
    inp = keras.Input((L, F))
    resid = layers.Flatten()(inp)
    forecast = None
    for b in range(n_blocks):
        h = resid
        for _ in range(n_layers):
            h = layers.Dense(width, activation="relu")(h)
        back = layers.Dense(L * F)(h)
        fc = layers.Dense(2 * H)(h)
        resid = layers.Subtract()([resid, back])
        forecast = fc if forecast is None else layers.Add()([forecast, fc])
    out = layers.Activation("softplus")(forecast)
    return keras.Model(inp, out, name="nbeats")


class KerasForecaster(Model):
    global_model = True

    def __init__(self, kind="lstm", epochs=30, batch_size=256, lr=1e-3, seed=0, verbose=0, patience=4,
                 target_clip=60.0):
        # target_clip: rare huge payments (>60x the usual daily flow) are winsorized
        # during *training only* so a handful of spikes do not dominate the MSE loss
        self.target_clip = target_clip
        self.kind, self.epochs, self.batch_size, self.lr = kind, epochs, batch_size, lr
        self.seed, self.verbose, self.patience = seed, verbose, patience
        self.name = {"lstm": "LSTM", "gru": "GRU", "nbeats": "N-BEATS"}[kind]
        self.net = None
        self.history = None

    def _build(self, L, F, H):
        if self.kind == "nbeats":
            return build_nbeats(L, F, H)
        return build_rnn(L, F, H, cell=self.kind)

    def fit(self, W: Windows, S: Series | None = None, W_val: Windows | None = None):
        keras = _keras()
        keras.utils.set_random_seed(self.seed)
        N, L, F = W.X.shape
        H = W.y.shape[1] // 2
        self.net = self._build(L, F, H)
        self.net.compile(optimizer=keras.optimizers.Adam(self.lr, clipnorm=1.0), loss="mse")
        cb = []
        val = None
        if W_val is not None and len(W_val):
            val = (W_val.X, np.minimum(W_val.y, self.target_clip))
            cb.append(keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True))
            cb.append(keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5))
        self.history = self.net.fit(W.X, np.minimum(W.y, self.target_clip), validation_data=val, epochs=self.epochs,
                                    batch_size=self.batch_size, callbacks=cb, verbose=self.verbose,
                                    shuffle=True)
        return self

    def predict(self, W: Windows, S: Series | None = None):
        H = W.y.shape[1] // 2
        p = self.net.predict(W.X, batch_size=2048, verbose=0) * W.scale[:, None]
        return p[:, :H], p[:, H:]

    def save(self, path):
        # re-compile with a fresh optimizer so the Adam state (2x the weights) is not
        # written to disk -- only needed for prediction, keeps files small for GitHub
        keras = _keras()
        self.net.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")
        self.net.save(path)

    @classmethod
    def load(cls, path, kind="gru"):
        keras = _keras()
        m = cls(kind=kind)
        m.net = keras.models.load_model(path, compile=False)  # inference only
        return m
