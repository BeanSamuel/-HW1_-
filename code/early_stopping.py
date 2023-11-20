class EarlyStopping:
    def __init__(self, patience=10, verbose=True, enabled=True):
        self.patience = patience
        self.verbose = verbose
        self.enabled = enabled
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):

        if not self.enabled:
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
