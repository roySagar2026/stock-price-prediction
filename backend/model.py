"""
model.py
--------
Model definitions and ensemble logic.
Supports: LightGBM (primary), XGBoost, Logistic Regression baseline, LSTM.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed. pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = None


# ─────────────────────────────────────────────────────────────────────────────
# Tree model factories
# ─────────────────────────────────────────────────────────────────────────────

def build_lgbm(cfg: dict):
    if not HAS_LGB:
        raise ImportError("lightgbm is required. pip install lightgbm")
    return lgb.LGBMClassifier(
        n_estimators      = cfg.get("n_estimators", 1500),
        max_depth         = cfg.get("max_depth", 6),
        learning_rate     = cfg.get("learning_rate", 0.02),
        subsample         = cfg.get("subsample", 0.8),
        colsample_bytree  = cfg.get("colsample_bytree", 0.7),
        min_child_samples = cfg.get("min_child_samples", 30),
        reg_alpha         = cfg.get("reg_alpha", 0.1),
        reg_lambda        = cfg.get("reg_lambda", 1.0),
        class_weight      = "balanced",
        random_state      = 42,
        verbose           = -1,
        n_jobs            = -1,
    )


def build_xgboost(cfg: dict):
    if not HAS_XGB:
        raise ImportError("xgboost is required. pip install xgboost")
    return xgb.XGBClassifier(
        n_estimators    = cfg.get("n_estimators", 1000),
        max_depth       = cfg.get("max_depth", 6),
        learning_rate   = cfg.get("learning_rate", 0.02),
        subsample       = cfg.get("subsample", 0.8),
        colsample_bytree= cfg.get("colsample_bytree", 0.7),
        eval_metric     = "auc",
        random_state    = 42,
        verbosity       = 0,
        n_jobs          = -1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class StockLSTM(nn.Module):
        """
        2-layer LSTM + dropout + two FC layers.
        Architecture from the research notebook (§7).
        """

        def __init__(self, input_size: int, hidden: int = 128,
                     layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden, layers,
                dropout=dropout if layers > 1 else 0,
                batch_first=True,
            )
            self.drop = nn.Dropout(dropout)
            self.fc1  = nn.Linear(hidden, 64)
            self.fc2  = nn.Linear(64, 1)
            self.act  = nn.ReLU()
            self.sig  = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])   # last time step
            out = self.act(self.fc1(out))
            return self.sig(self.fc2(out)).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Soft-voting ensemble
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Soft-vote ensemble of any subset of {lgbm, xgb, lstm}.
    Weights can be adjusted to favour more accurate models.
    """

    def __init__(self, models: dict, weights: Optional[dict] = None):
        """
        models  : {"lgbm": trained_lgbm, "xgb": trained_xgb, ...}
        weights : {"lgbm": 0.5, "xgb": 0.3, "lstm": 0.2}
        """
        self.models = models
        n = len(models)
        self.weights = weights or {k: 1 / n for k in models}

    def predict_proba(self, X_tree, X_lstm=None) -> np.ndarray:
        """
        Returns probability of Class 1 (UP > threshold in N days).
        X_tree : 2-D numpy array for tree models
        X_lstm : 3-D numpy array (batch, seq_len, features) for LSTM
        """
        probas = {}

        for name, model in self.models.items():
            if name == "lstm":
                if X_lstm is None or not HAS_TORCH:
                    continue
                model.eval()
                with torch.no_grad():
                    t = torch.tensor(X_lstm, dtype=torch.float32).to(DEVICE)
                    p = model(t).cpu().numpy()
                probas["lstm"] = p
            else:
                probas[name] = model.predict_proba(X_tree)[:, 1]

        if not probas:
            raise RuntimeError("No models returned predictions.")

        total_weight = sum(self.weights.get(k, 0) for k in probas)
        blended = sum(
            probas[k] * self.weights.get(k, 0) / total_weight
            for k in probas
        )
        return blended
