import json
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from ml_project.enities.model_params import ModelParams

logger = logging.getLogger(__name__)


def get_model(params: ModelParams):
    if params.model == "LogReg":
        model = LogisticRegression(**params.logreg_params.__dict__)
        logger.info(msg="LogReg model loaded")

    elif params.model == "RandomForest":
        model = RandomForestClassifier(**params.forest_params.__dict__)
        logger.info(msg="RandomForest model loaded")
    else:
        raise ValueError(f"Unknown model: {params.model}")
    return model


def test_model(pipe: Pipeline, test_X: pd.DataFrame, test_y: pd.Series,
               ) -> dict:
    predicted_y = pipe.predict(test_X)
    metrics = {
        "r2_score": r2_score(test_y, predicted_y),
        "rmse": mean_squared_error(test_y, predicted_y, squared=False),
        "mae": mean_absolute_error(test_y, predicted_y),
    }
    return metrics


def save_metrics_to_json(metrics: dict, path: str) -> None:
    metrics_path = os.path.abspath(path)
    with open(metrics_path, "w+") as file:
        json.dump(metrics, file, indent=6)
        logger.info(msg="Metris saved")


def serialize_model(model: object, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
        logger.info(msg="Model serialized")


def open_model(path: str) -> LogisticRegression:
    with open(path, "rb") as f:
        model = pickle.load(f)
        logger.info(msg="Model opened")
    return model


def save_predict(predict: np.ndarray, path: str):
    predict_df = pd.DataFrame({'DEF': predict})
    predict_df.to_csv(os.path.abspath(path), index=False)
    logger.info(msg="Predict saved")
