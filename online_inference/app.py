import logging
import os
import sys
import uvicorn

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from online_inference.data_utils import (
    InputData,
    OutputData,
    ModelType,
    get_data,
    get_model,
)

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


app = FastAPI()

model_logreg = None
model_forest = None


@app.get("/")
def main():
    return "predictor is alive :)"


@app.on_event("startup")
def startup():
    global model_logreg, model_forest
    model_logreg_path = os.getenv("PATH_TO_MODEL_LR")
    if model_logreg_path is None:
        logger.error(f"PATH_TO_MODEL_LR not found")
        raise RuntimeError(f"PATH_TO_MODEL_LR not found")

    model_forest_path = os.getenv("PATH_TO_MODEL_RF")
    if model_forest_path is None:
        logger.error(f"PATH_TO_MODEL_RF not found")
        raise RuntimeError(f"PATH_TO_MODEL_RF not found")

    model_logreg = get_model(model_logreg_path)
    model_forest = get_model(model_forest_path)
    logger.info(msg="Models loaded")


@app.post("/predict", response_model=OutputData)
def predict(request: InputData):
    data = get_data(request)

    logger.info(msg=f"Data loaded")
    logger.info(msg=f"Predict using {request.model} model")

    if request.model == ModelType.logreg:
        model = model_logreg
    elif request.model == ModelType.forest:
        model = model_forest
    else:
        logger.error(msg=f"Unknown model type")
        raise HTTPException(status_code=400, detail="Unknown model type")

    try:
        y_pred = model.predict(data)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error: smth went wrong while prediction")

    logger.info(msg=f"Prediction finished. It's OK :) {y_pred}")
    return OutputData(predicted_values=[str(pred) for pred in y_pred])


@app.get("/health")
def health():
    if model_logreg is None or model_forest is None:
        raise HTTPException(status_code=500, detail="No models found :(!")
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"detail": "Models loaded successfully :)"}),
    )


@app.exception_handler(RequestValidationError)
async def validate_data(
    _: Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
