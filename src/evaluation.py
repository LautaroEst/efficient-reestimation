
from .expected_cost.psrcal_wrappers import Brier, LogLoss
from sklearn.metrics import accuracy_score



metric_name2fn = {
    "LogLoss": LogLoss,
    "Brier": Brier
}


def compute_score(true_labels, predictions, score="accuracy"):
    if score == "accuracy":
        return accuracy_score(true_labels, predictions)
    else:
        raise ValueError(f"Score {score} not supported!")



