class Metric:
    """Store the average and current value for a set of metrics.
    """
    def update(self, y_true, y_pred):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def reset(self):
        pass
