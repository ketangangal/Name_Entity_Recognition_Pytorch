from transformers import TrainingArguments
from transformers import Trainer
from transformers import DataCollatorForTokenClassification
from src.ner_model_evaulator.performace import compute_metrics
from src.ner_model.model import XLMRobertaForTokenClassification


class TrainTokenClassifier:
    def __init__(self, tokenizer, data):
        self.num_epochs = 1
        self.batch_size = 512
        self.data = data
        self.logging_steps = len(self.data["train"]) // self.batch_size
        self.model_name = f"{self.data}-finetuned-panx-de"
        self.xlmr_tokenizer = tokenizer
        self.xlmr_model_name = "Name"
        self.xlmr_config = "Name"

    def Create_training_args(self):

        training_args = TrainingArguments(
            output_dir=self.model_name,
            log_level="error",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            save_steps=1e6,
            weight_decay=0.01,
            disable_tqdm=False,
            logging_steps=self.logging_steps,
            push_to_hub=False)

        return training_args

    def model_init(self):
        return XLMRobertaForTokenClassification.from_pretrained(self.xlmr_model_name, config=self.xlmr_config)

    def data_collector(self):
        return DataCollatorForTokenClassification(self.xlmr_tokenizer)

    def train(self):
        trainer = Trainer(model_init=self.model_init,
                          args=self.Create_training_args(),
                          data_collator=self.data_collector(),
                          compute_metrics=compute_metrics,
                          train_dataset=self.data["train"],
                          eval_dataset=self.data["validation"],
                          tokenizer=self.xlmr_tokenizer)

        trainer.train()