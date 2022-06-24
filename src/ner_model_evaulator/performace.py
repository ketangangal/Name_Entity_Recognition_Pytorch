import numpy as np
from seqeval.metrics import f1_score


class Performance:
    def __init__(self, index2tag):
        self.index2tag = index2tag

    def align_predictions(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                # Ignore label IDs = -100
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(self.index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(self.index2tag[preds[batch_idx][seq_idx]])

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list

    def compute_metrics(self, eval_pred):
        y_pred, y_true = self.align_predictions(eval_pred.predictions,
                                           eval_pred.label_ids)

        return {"f1": f1_score(y_true, y_pred)}