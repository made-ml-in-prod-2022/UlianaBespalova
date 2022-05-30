import boto3
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple

from ml_project.enities import DatasetParams, SplittingParams

logger = logging.getLogger(__name__)


def download_dataset(s3_bucket: str, s3_access: str, output: str):
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=s3_access['endpoint_url'],
        aws_access_key_id=s3_access['pk'],
        aws_secret_access_key=s3_access['sk']
    )
    s3.download_file(s3_bucket, s3_access['path'], output)


def read_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(os.path.abspath(path))
        logger.info(msg="Dataset loaded")
    except IOError as e:
        logger.error(msg="Error: can't load dataset")
        raise e
    return df


def split_train_val_data(df: pd.DataFrame, splitting_params: SplittingParams) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(msg="Dataset splitted")

    return train_test_split(
        df,
        test_size=splitting_params.val_size,
        random_state=splitting_params.random_state,
    )


def get_data(params: DatasetParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_dataset(params.data_path)
    return split_train_val_data(df, params.splitting_params)
