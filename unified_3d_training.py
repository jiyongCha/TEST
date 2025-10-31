"""Unified 3D U-Net training utilities.

This module contains utilities for loading thermal / power CSV data and
training a 3D U-Net model that fuses geometric parameters.  The
implementation is based on the script provided in the prompt, but the
power map loader has been hardened so that it can discover power maps in
case-specific subdirectories instead of silently falling back to thermal
maps, which previously prevented the expected improvements when power
maps were present.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv3D,
    Dense,
    Dropout,
    Input,
    Lambda,
    MaxPooling3D,
    Reshape,
    UpSampling3D,
)
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Third-party metrics -----------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score as sklearn_r2_score

# Regular expressions -----------------------------------------------------
PARAM_RE = re.compile(
    r"^HBM3_Die_(?P<die>[^_]+)_GH_(?P<gap>[^_]+)_sub_(?P<kd>[^_]+)_(?P<ks>[^_]+)_(?P<kE>[^_]+)_(?P<P>[^_]+)$"
)


# Utility dataclasses -----------------------------------------------------
@dataclass
class DatasetCase:
    """Lightweight container describing a single simulation case."""

    name: str
    thermal_csv_dir: str


# Metric helpers ----------------------------------------------------------
def r2_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Keras-compatible R² metric.

    The previous implementation computed the entire metric within a single
    TensorFlow graph call, which is kept here.  The metric is registered so
    it can be tracked during training.
    """

    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - ss_res / (ss_tot + K.epsilon())


# Dataset helpers ---------------------------------------------------------
def parse_params_from_name(name: str) -> np.ndarray:
    match = PARAM_RE.match(name)
    if not match:
        raise ValueError(f"Invalid case folder name: {name}")
    vals = match.groupdict()
    return np.array(
        [
            float(vals["die"]),
            float(vals["gap"]),
            float(vals["kd"]),
            float(vals["ks"]),
            float(vals["kE"]),
            float(vals["P"]),
        ],
        dtype=np.float32,
    )


def list_case_dirs(root: str) -> List[DatasetCase]:
    cases: List[DatasetCase] = []
    for entry in sorted(os.listdir(root)):
        csv_dir = os.path.join(root, entry, "csvFile")
        if os.path.isdir(csv_dir) and entry.startswith("HBM3_Die_"):
            cases.append(DatasetCase(name=entry, thermal_csv_dir=csv_dir))
    if not cases:
        raise RuntimeError(f"No case directories found under {root}")
    return cases


def load_thermal_volume(case_csv_dir: str) -> np.ndarray:
    base = np.loadtxt(os.path.join(case_csv_dir, "base.csv"), delimiter=",", dtype=np.float32)
    depth_slices = [base]
    for core_idx in range(1, 13):
        depth_slices.append(
            np.loadtxt(
                os.path.join(case_csv_dir, f"core{core_idx}.csv"),
                delimiter=",",
                dtype=np.float32,
            )
        )
    return np.stack(depth_slices, axis=0)


def _candidate_power_dirs(power_root: str, case_name: str) -> Iterable[str]:
    """Yield candidate directories that may contain power maps for *case_name*.

    The original implementation only checked the bare ``power_root``
    directory, which caused the loader to fall back to thermal maps for
    every case whenever the power maps were stored in
    ``power_root/<case>/`` directories.  The loader now tries the most
    common directory layouts before giving up.
    """

    # Case-specific folder mirroring the thermal directory structure.
    yield os.path.join(power_root, case_name, "csvFile")
    yield os.path.join(power_root, case_name)

    # Flat directory layout (legacy behaviour).
    yield power_root

    # Nested directories where files are grouped by the P value.  The P
    # component is the substring after the last underscore in the case name.
    match = PARAM_RE.match(case_name)
    if match:
        p_component = match.group("P")
        yield os.path.join(power_root, p_component)


def _load_power_map(path: str, expected_shape: Tuple[int, int]) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception:
        return None
    if data.shape != expected_shape:
        return None
    return data


def try_load_power_maps(
    power_root: str,
    case_name: str,
    H: int,
    W: int,
    P_val_str: str,
) -> List[np.ndarray]:
    """Attempt to load base + 12 core power maps for ``case_name``.

    The loader now searches multiple directory layouts.  If *any* map is
    missing or has an unexpected shape the function returns an empty list
    so the caller can fall back to the thermal volume.  Logging is kept
    concise to avoid overwhelming the console with repeated warnings.
    """

    expected_shape = (H, W)
    filenames = [f"power_base_{P_val_str}.csv"] + [f"power_core{i}_{P_val_str}.csv" for i in range(1, 13)]

    for candidate_dir in _candidate_power_dirs(power_root, case_name):
        maps: List[np.ndarray] = []
        missing: List[str] = []
        for fname in filenames:
            arr = _load_power_map(os.path.join(candidate_dir, fname), expected_shape)
            if arr is None:
                missing.append(fname)
                break
            maps.append(arr)
        if maps and not missing:
            print(f"[INFO] Loaded power maps from {candidate_dir}")
            return maps
        if missing and maps:
            # Some files were found but at least one failed validation.  Try next layout.
            print(
                f"[WARN] Partial power maps for {case_name} in {candidate_dir}; missing or invalid: {', '.join(missing)}"
            )
    print(f"[INFO] Falling back to thermal volume for {case_name}; no matching power maps found.")
    return []


# Parameter scaling -------------------------------------------------------
class ParamScaler:
    def __init__(self) -> None:
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None

    def fit(self, params: np.ndarray) -> None:
        self.min = params.min(axis=0)
        self.max = params.max(axis=0)
        self.max = np.where(self.max == self.min, self.min + 1.0, self.max)

    def transform(self, params: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise RuntimeError("ParamScaler must be fitted before calling transform().")
        return (params - self.min) / (self.max - self.min)

    def fit_transform(self, params: np.ndarray) -> np.ndarray:
        self.fit(params)
        return self.transform(params)


# Dataset builder ---------------------------------------------------------
def build_dataset_3d(
    thermal_root: str,
    power_root: str,
    use_power_channels: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    P_list: List[np.ndarray] = []

    for case in list_case_dirs(thermal_root):
        params = parse_params_from_name(case.name)
        thermal_vol = load_thermal_volume(case.thermal_csv_dir)
        depth, H_full, W_full = thermal_vol.shape

        match = PARAM_RE.match(case.name)
        if not match:
            raise ValueError(f"Invalid case folder name (re-check): {case.name}")
        P_val_str = match.group("P")

        power_maps = try_load_power_maps(power_root, case.name, H_full, W_full, P_val_str)
        if power_maps:
            spatial = np.stack(power_maps, axis=0)
        else:
            spatial = thermal_vol

        X_full = np.repeat(spatial[..., np.newaxis], use_power_channels, axis=-1)
        Y_full = thermal_vol[..., np.newaxis]

        Y_cropped = Y_full[:, 4:92, 2:110, :]
        X_cropped = X_full[:, 4:92, 2:110, :]

        if Y_cropped.shape[1] == 0 or Y_cropped.shape[2] == 0:
            print(
                f"[WARN] Cropping removed all data for {case.name} (original width {W_full}); skipping case."
            )
            continue

        X_list.append(X_cropped)
        Y_list.append(Y_cropped)
        P_list.append(params)

    if not X_list:
        raise RuntimeError("No valid data loaded. Check cropping dimensions and file paths.")

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    P = np.stack(P_list).astype(np.float32)
    print(f"Built dataset: X{X.shape}, Y{Y.shape}, P{P.shape}")
    return X, Y, P


# Model -------------------------------------------------------------------
def conv3d_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = Conv3D(filters, (3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(filters, (3, 3, 3), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def build_3d_unet_with_params(input_shape: Tuple[int, int, int, int], param_dim: int) -> Model:
    img_in = Input(shape=input_shape, name="image_input")

    e1 = conv3d_block(img_in, 16)
    p1 = MaxPooling3D((1, 2, 2))(e1)

    e2 = conv3d_block(p1, 32)
    p2 = MaxPooling3D((1, 2, 2))(e2)

    e3 = conv3d_block(p2, 64)

    par_in = Input(shape=(param_dim,), name="param_input")
    d1 = Dense(64, activation="relu")(par_in)
    d2 = Dense(128, activation="relu")(d1)
    param_expanded = Reshape((1, 1, 1, 128))(d2)

    def tile_for_concat(inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        e3_tensor, param_tensor = inputs
        shape = tf.shape(e3_tensor)
        d_shape, h_shape, w_shape = shape[1], shape[2], shape[3]
        return tf.tile(param_tensor, [1, d_shape, h_shape, w_shape, 1])

    param_tiled = Lambda(tile_for_concat, name="tile_parameters")([e3, param_expanded])
    fused_bottleneck = Concatenate()([e3, param_tiled])
    bottleneck = Conv3D(64, (1, 1, 1), padding="same", activation="relu")(fused_bottleneck)

    u1 = UpSampling3D((1, 2, 2))(bottleneck)
    d1 = Concatenate()([u1, e2])
    d1 = conv3d_block(d1, 32)
    d1 = Dropout(0.2)(d1)

    u2 = UpSampling3D((1, 2, 2))(d1)
    d2 = Concatenate()([u2, e1])
    d2 = conv3d_block(d2, 16)

    out = Conv3D(1, (1, 1, 1), activation="linear", padding="same", name="thermal_out")(d2)

    model = Model(inputs=[img_in, par_in], outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mse",
        metrics=["mse", "mae", RootMeanSquaredError(name="rmse"), r2_score],
    )
    return model


# Training ----------------------------------------------------------------
def train_unified_3d(
    thermal_root: str,
    power_root: str,
    epochs: int = 300,
    batch_size: int = 2,
    val_split: float = 0.1,
    use_power_channels: int = 3,
    scale_params: bool = True,
    output_dir: str | None = None,
) -> dict:
    X, Y, P = build_dataset_3d(thermal_root, power_root, use_power_channels)

    Y_min = Y.min()
    Y_max = Y.max()
    Y_scale = Y_max - Y_min + 1e-7
    Y = (Y - Y_min) / Y_scale

    X_min = X.min()
    X_max = X.max()
    X_scale = X_max - X_min + 1e-7
    X = (X - X_min) / X_scale

    if scale_params:
        scaler = ParamScaler()
        Pn = scaler.fit_transform(P)
    else:
        scaler = None
        Pn = P

    model = build_3d_unet_with_params(X.shape[1:], Pn.shape[1])

    if output_dir is None:
        output_dir = os.path.join(thermal_root, "unified_3d_results")
    os.makedirs(output_dir, exist_ok=True)

    scaling_info = {
        "Y_min": float(Y_min),
        "Y_max": float(Y_max),
        "Y_scale": float(Y_scale),
        "X_min": float(X_min),
        "X_max": float(X_max),
        "X_scale": float(X_scale),
    }
    if scaler is not None:
        scaling_info["P_min"] = scaler.min.tolist()
        scaling_info["P_max"] = scaler.max.tolist()
    with open(os.path.join(output_dir, "scaling_info.json"), "w", encoding="utf-8") as fh:
        json.dump(scaling_info, fh, indent=2)

    csv_logger = CSVLogger(os.path.join(output_dir, "fit_log.csv"))

    history = model.fit(
        [X, Pn],
        Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[
            ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, verbose=1),
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
            csv_logger,
        ],
        verbose=1,
    )

    model.save(os.path.join(output_dir, "unet3d_unified.keras"))

    Y_pred = model.predict([X, Pn], batch_size=batch_size)
    Y_flat = Y.reshape(-1)
    Y_pred_flat = Y_pred.reshape(-1)
    mse_norm = mean_squared_error(Y_flat, Y_pred_flat)
    mae_norm = mean_absolute_error(Y_flat, Y_pred_flat)
    rmse_norm = float(np.sqrt(mse_norm))
    r2_norm = float(sklearn_r2_score(Y_flat, Y_pred_flat))

    metrics = {
        "mse_normalized": float(mse_norm),
        "mae_normalized": float(mae_norm),
        "rmse_normalized": rmse_norm,
        "r2_score_normalized": r2_norm,
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


__all__ = [
    "DatasetCase",
    "ParamScaler",
    "build_dataset_3d",
    "build_3d_unet_with_params",
    "train_unified_3d",
    "try_load_power_maps",
]
