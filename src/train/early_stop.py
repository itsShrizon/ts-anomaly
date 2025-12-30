class EarlyStop:
    def __init__(self, patience: int = 6, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best = float("inf")
        self.bad = 0
        self.stop = False

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best, self.bad = metric, 0
        else:
            self.bad += 1
            if self.bad >= self.patience:
                self.stop = True
        return self.stop
