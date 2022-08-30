from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.config.configurations import Configuration
from ner.exception.exception import CustomException
from typing import Any, Dict, List, ClassVar
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import torch
import os
import sys

app = FastAPI()
class PredictPipeline:
    def __init__(self, config):
        self.predict_pipeline_config = config.get_model_predict_pipeline_config()
        self.tokenizer = self.predict_pipeline_config.tokenizer

        if len(os.listdir(self.predict_pipeline_config.output_dir)) == 0:
            raise LookupError("Model not found : please Run Training Pipeline from pipeline/train_pipeline.py")

        self.model = XLMRobertaForTokenClassification.from_pretrained(self.predict_pipeline_config.output_dir)

    def run_data_preparation(self, data: str):
        try:
            data = data.split()
            input_ids = self.tokenizer(data, truncation=self.predict_pipeline_config.truncation,
            is_split_into_words=self.predict_pipeline_config.is_split_into_words)
            formatted_data = torch.tensor(input_ids["input_ids"]).reshape(-1, 1)
            outputs = self.model(formatted_data).logits
            predictions = torch.argmax(outputs, dim=-1)
            pred_tags = [self.predict_pipeline_config.index2tag[i.item()] for i in predictions[1:-1]]
            return pred_tags[1:-1]
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, data):
        predictions = self.run_data_preparation(data)
        response = {
            "Input_Data": data.split(),
            "Tags": predictions
        }
        print(response)
        return response

pipeline = PredictPipeline(Configuration())

@app.get("/train")
@app.post("/train")
def train(request: Request):
    if request.method == "GET":
        train_info = {"Pipeline": "To train please use POST method",
                      "Metadata": "Created using fastapi"}
        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    elif request.method == "POST":

        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    else:
        JSONResponse(content={"Error": True}, status_code=400, media_type="application/json")


@app.get("/predict")
@app.post("/predict/{data}")
def predict(request: Request,data:str):
    if request.method == "GET":
        train_info = {"Pipeline": "To Predict please use POST method",
                      "Metadata": "Created using fastapi"}
        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    elif request.method == "POST":
        response = pipeline.run_pipeline(data)
        return JSONResponse(content=response, status_code=200, media_type="application/json")

    else:
        JSONResponse(content={"Error": True}, status_code=400, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8085, reload=True)
