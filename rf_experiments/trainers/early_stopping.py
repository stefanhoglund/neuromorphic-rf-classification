import copy
import numpy as np

class EarlyStopping:
    def __init__(self, mode="min", patience=5, min_delta=0.0):
        """
        mode: "min" for loss, "max" for accuracy/F1/etc.
        patience: how many epochs with no improvement before stopping
        min_delta: minimum change to count as an improvement
        """
        assert mode in ["min", "max"]
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False
        self.best_state = None  # optional: to restore best weights

    def update(self, score, model=None):
        """
        score: current validation metric (loss or accuracy)
        model: optional model to snapshot when improvement happens
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
            return

        improve = False
        if self.mode == "min":
            improve = score < self.best_score - self.min_delta
        else:
            improve = score > self.best_score + self.min_delta

        if improve:
            self.best_score = score
            self.num_bad_epochs = 0
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True

