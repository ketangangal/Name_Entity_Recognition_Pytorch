from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.config.configurations import Configuration
from ner.exception.exception import CustomException
from typing import Any, Dict, List, ClassVar
import torch
import logging
import sys

logger = logging.getLogger(__name__)


class PredictPipeline:
    def __init__(self, config):
        self.config = config

    def run_data_preparation(self, data: str):
        try:
            data = data.split()
            predict_pipeline_config = self.config.get_model_predict_pipeline_config()

            tokenizer = predict_pipeline_config.tokenizer

            input_ids = tokenizer(data, truncation=predict_pipeline_config.truncation,
                                  is_split_into_words=predict_pipeline_config.is_split_into_words)
            formatted_data = torch.tensor(input_ids["input_ids"]).reshape(-1, 1)
            model = XLMRobertaForTokenClassification.from_pretrained(predict_pipeline_config.output_dir)
            outputs = model(formatted_data).logits
            predictions = torch.argmax(outputs, dim=-1)
            pred_tags = [predict_pipeline_config.index2tag[i.item()] for i in predictions[1:-1]]
            return pred_tags[1:-1]
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)

    def run_pipeline(self, data):
        predictions = self.run_data_preparation(data)
        response = {
            "Input_Data": data.split(),
            "Tags": predictions
        }
        print(response)
        return response


if __name__ == "__main__":
    pipeline = PredictPipeline(Configuration())
    pipeline.run_pipeline("ineuron in usa")
