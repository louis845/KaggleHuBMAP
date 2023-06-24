# utils class for image model training.

class ImageMetrics:
    def __init__(self, name: "str", cum_metrics: list[str]):
        self.name = name
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

        self.history = []

        self.cum_metrics = {}
        for metric in cum_metrics:
            self.cum_metrics[metric] = 0

    def record_metric(self, true_positive:int, false_positive:int, true_negative:int, false_negative:int, metrics:dict[str, float]):
        # assert the keys of metrics are the same as cum_metrics
        assert set(metrics.keys()) == set(self.cum_metrics.keys())

        self.true_positive += true_positive
        self.false_positive += false_positive
        self.true_negative += true_negative
        self.false_negative += false_negative

        for metric in metrics:
            self.cum_metrics[metric] += metrics[metric]

    def summarize_metrics(self, epoch_batch_size: int):
        if self.true_positive + self.false_negative == 0:
            recall = 0
        else:
            recall = self.true_positive / (self.true_positive + self.false_negative)

        if self.true_positive + self.false_positive == 0:
            precision = 0
        else:
            precision = self.true_positive / (self.true_positive + self.false_positive)

        accuracy = (self.true_positive + self.true_negative) / (self.true_positive + self.true_negative + self.false_positive + self.false_negative)

        metrics_dict = {"recall": recall, "precision": precision, "accuracy": accuracy}
        for metric in self.cum_metrics:
            metrics_dict[metric] = self.cum_metrics[metric] / epoch_batch_size

        self.history.append(metrics_dict)

        self.true_positive, self.false_positive, self.true_negative, self.false_negative = 0, 0, 0, 0
        for metric in self.cum_metrics:
            self.cum_metrics[metric] = 0

    def print_latest_metrics(self):
        print()