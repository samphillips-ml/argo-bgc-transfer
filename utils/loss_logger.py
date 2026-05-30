"""
utils/loss_logger.py

Simple CSV loss logger. Each training stage writes its own file under results/.
Default columns: epoch, train_loss, val_loss
Extra columns can be added via the `extras` parameter for stages with
multiple loss terms (e.g. joint finetuning).
"""

import os
import csv


class LossLogger:

    def __init__(self, path, extras=None):
        """
        path   : output CSV path
        extras : list of additional column names beyond train_loss/val_loss
                 e.g. ["ts_raw", "ts_evo", "oxy"]
        """
        self.path    = path
        self.extras  = extras or []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"] + self.extras)

    def log(self, epoch, train_loss, val_loss, **kwargs):
        """
        epoch      : int
        train_loss : float
        val_loss   : float
        kwargs     : values for any extra columns declared at init
        """
        extra_vals = [f"{kwargs[k]:.6f}" if isinstance(kwargs[k], float) else str(kwargs[k]) for k in self.extras if k in kwargs]
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"] + extra_vals)