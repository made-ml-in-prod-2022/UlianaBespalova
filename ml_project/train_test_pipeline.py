import logging
import sys
import click
from sklearn.pipeline import Pipeline

from data import get_data, read_dataset
from enities import get_train_params, get_predict_params
from features import Transformer, extract_target
from models import (
    test_model,
    get_model,
    save_metrics_to_json,
    serialize_model,
    open_model,
    save_predict,
)


def add_logger():
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger


logger = add_logger()


def train_model_pipeline(config_path):
    logger.info(msg="*** Training pipeline is started ***")

    params = get_train_params(config_path)
    train_df, test_df = get_data(params.data)
    logger.info(msg="Got data")

    train_X, train_y, test_X, test_y = extract_target(train_df, test_df, params.features.target)
    transformer = Transformer(params.features)
    model = get_model(params.model)
    logger.info(msg="Load model")

    pipeline = Pipeline([("transformer", transformer), ("model", model)])
    pipeline.fit(train_X, train_y)
    logger.info(msg="Pipeline fitted")

    res_metrics = test_model(pipeline, test_X, test_y)
    save_metrics_to_json(res_metrics, params.model.metric_path)
    logger.info(msg="Metrics saved")

    serialize_model(pipeline, params.model.save_path)
    logger.info(msg="Model saved")
    logger.info(msg="*** Training pipeline is finished. It's OK :) ***")


def predict_model_pipeline(config_path):
    logger.info(msg="*** Predicting pipeline is started ***")

    params = get_predict_params(config_path)
    df = read_dataset(params.data_path)
    logger.info(msg="Got data")

    model = open_model(params.model)
    logger.info(msg="Load model")

    predicted_y = model.predict(df)
    logger.info(msg="Results predicted")
    save_predict(predicted_y, params.results_path)
    logger.info(msg="Results saved")

    logger.info(msg="*** Predicting pipeline is finished. It's OK :) ***")


@click.command()
@click.option(
    "--config",
    required=True,
    type=str
)
@click.option(
    "--mode",
    required=True,
    type=str
)
def main(mode, config):
    if mode == "fit":
        train_model_pipeline(config)
    elif mode == "predict":
        predict_model_pipeline(config)
    else:
        raise ValueError("Error: Unknown mode")


if __name__ == "__main__":
    main()
