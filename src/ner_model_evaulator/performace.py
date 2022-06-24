import numpy as np
from seqeval.metrics import f1_score


def align_predictions(predictions, label_ids, index2tag):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # Ignore label IDs = -100
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list


def compute_metrics(eval_pred, index2tag):
    y_pred, y_true = align_predictions(eval_pred.predictions,
                                       eval_pred.label_ids, index2tag)
    return {"f1": f1_score(y_true, y_pred)}