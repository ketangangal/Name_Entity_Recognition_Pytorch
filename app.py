from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


@app.get("/train")
@app.post("/train")
def train(request: Request):
    if request.method == "GET":
        train_info = {"Pipeline": "Requires large Engine to train",
                      "Metadata": "Please check out train.py in ner package"}
        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    elif request.method == "POST":
        train_info = {"Pipeline": "Pipeline triggered",
                      "Metadata": "You will get trained model in the memory Thanks"}
        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    else:
        JSONResponse(content={"Error": True}, status_code=400, media_type="application/json")


@app.get("/predict")
@app.post("/predict")
def predict(request: Request):
    if request.method == "GET":
        train_info = {"Pipeline": "Requires large Engine to train",
                      "Metadata": "Please check out train.py in ner package"}
        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    elif request.method == "POST":
        train_info = {"Pipeline": "Pipeline triggered",
                      "Metadata": "You will get trained model in the memory Thanks"}
        return JSONResponse(content=train_info, status_code=200, media_type="application/json")

    else:
        JSONResponse(content={"Error": True}, status_code=400, media_type="application/json")


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8085, reload=True)
