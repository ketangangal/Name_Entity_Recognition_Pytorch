from transformers import Trainer
from ner.exception.exception import CustomException
from transformers import TrainingArguments
from ner.components.model_architecture import XLMRobertaForTokenClassification
from transformers import DataCollatorForTokenClassification
from ner.entity.config_entity import ModelTrainConfig
from typing import Any, Dict, AnyStr
import logging
import sys
import numpy as np
from seqeval.metrics import f1_score

logger = logging.getLogger(__name__)


class TrainTokenClassifier:
    def __init__(self, model_training_config: ModelTrainConfig, processed_data: Dict):
        self.model_training_config = model_training_config
        self.processed_data = processed_data

    def create_training_args(self):
        try:
            logging_steps = len(self.processed_data["train"].select(range(100))) // self.model_training_config.batch_size
            
            training_args = TrainingArguments(
                output_dir=self.model_training_config.output_dir,
                log_level="error",
                num_train_epochs=self.model_training_config.epochs,
                per_device_train_batch_size=self.model_training_config.batch_size,
                per_device_eval_batch_size=self.model_training_config.batch_size,
                save_steps=self.model_training_config.save_steps,
                weight_decay=0.01,
                disable_tqdm=False,
                logging_steps=logging_steps,
                push_to_hub=False)

            return training_args
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def model_init(self):
        try:
            return XLMRobertaForTokenClassification.from_pretrained(self.model_training_config.model_name,
                                                                    config=self.model_training_config.xlmr_config)
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def data_collector(self):
        try:
            return DataCollatorForTokenClassification(self.model_training_config.tokenizer)
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def align_predictions(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        labels_list, preds_list = [], []

        for batch_idx in range(batch_size):
            example_labels, example_preds = [], []
            for seq_idx in range(seq_len):
                # Ignore label IDs = -100
                if label_ids[batch_idx, seq_idx] != -100:
                    example_labels.append(self.model_training_config.index2tag[label_ids[batch_idx][seq_idx]])
                    example_preds.append(self.model_training_config.index2tag[preds[batch_idx][seq_idx]])

            labels_list.append(example_labels)
            preds_list.append(example_preds)

        return preds_list, labels_list

    def compute_metrics(self, eval_pred):
        y_pred, y_true = self.align_predictions(eval_pred.predictions, eval_pred.label_ids)
        return {"f1": f1_score(y_true, y_pred)}

    def train(self):
        try:
            logger.info(" Training Started ")
            trainer = Trainer(model_init=self.model_init,
                              args=self.create_training_args(),
                              data_collator=self.data_collector(),
                              compute_metrics=self.compute_metrics,
                              train_dataset=self.processed_data["train"].select(range(100)),
                              eval_dataset=self.processed_data["validation"].select(range(10)),
                              tokenizer=self.model_training_config.tokenizer)

            logger.info(" Training Running ")
            result = trainer.train()
            logger.info(f" Result of the training {result} ")
            trainer.save_model(self.model_training_config.output_dir)
            logger.info(f" Model Saved at [{self.model_training_config.output_dir}]")
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

