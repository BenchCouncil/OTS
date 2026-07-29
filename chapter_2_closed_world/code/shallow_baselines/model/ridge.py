from __future__ import annotations

from .ols import OLSForecaster


class RidgeForecaster(OLSForecaster):
    """Temporal linear forecaster with an explicit ridge penalty."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        fit_intercept: bool = True,
        ridge_alpha: float = 1.0,
        rcond: float = 1e-10,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha if ridge_alpha > 0 else 1.0,
            rcond=rcond,
        )
