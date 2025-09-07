import torch
import torch.nn.functional as F

@torch.no_grad()
def binary_classification_metrics(logits, targets, threshold=0.5, eps=1e-12):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    tp = ((preds == 1) & (targets == 1)).sum().float()
    tn = ((preds == 0) & (targets == 0)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # AUROC/AUPRC via torchmetrics? Not available here; implement simple ROC AUC approximation if needed
    acc = (preds == targets).float().mean()
    return {
        "acc": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "tp": tp.item(),
        "tn": tn.item(),
        "fp": fp.item(),
        "fn": fn.item(),
    }
