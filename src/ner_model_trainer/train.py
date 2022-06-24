from transformers import Trainer
from src.ner_utils.utils import read_config
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from src.ner_model.model import XLMRobertaForTokenClassification
from src.ner_data_ingestion.data_ingestion import DataIngestion
from src.ner_data_prepration.data_prepration import Preprocessing
from src.ner_model_evaulator.performace import Performance

config = read_config()


class TrainTokenClassifier:
    def __init__(self, tokenizer, data, xlmr_config, index2tag):
        self.data = data
        self.xlmr_tokenizer = tokenizer
        self.xlmr_config = xlmr_config
        self.num_epochs = config["Train_Params"]["epochs"]
        self.batch_size = config["Train_Params"]["batch_size"]
        self.logging_steps = len(self.data["train"]) // self.batch_size
        self.xlmr_model_name = config["Train_Params"]["model_name"]
        self.model_name = f"{self.xlmr_model_name}-finetuned-panx-de"
        self.metrics = Performance(index2tag)

    def Create_training_args(self):
        training_args = TrainingArguments(
            output_dir=self.model_name,
            log_level="error",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="epoch",
            save_steps=1000000,
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
        print("Training Started")
        trainer = Trainer(model_init=self.model_init,
                          args=self.Create_training_args(),
                          data_collator=self.data_collector(),
                          compute_metrics=self.metrics.compute_metrics,
                          train_dataset=self.data["train"],
                          eval_dataset=self.data["validation"],
                          tokenizer=self.xlmr_tokenizer)

        print(trainer.train())


if __name__ == "__main__":
    ingest = DataIngestion()
    data = ingest.get_data()

    prep = Preprocessing(data)
    panx_en_encoded, index2tag, tag2index = prep.prepare_data_for_fine_tuning()
    print(panx_en_encoded)
    model = TrainTokenClassifier(prep.tokenization(), panx_en_encoded, prep.create_auto_config(), index2tag)
    model.train()


