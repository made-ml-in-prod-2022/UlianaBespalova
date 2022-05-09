import logging
import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from enities import FeatureParams


logger = logging.getLogger(__name__)


def extract_target(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    train_X = train_df.drop(target, axis=1, inplace=False)
    train_y = train_df[target]
    test_X = test_df.drop(target, axis=1, inplace=False)
    test_y = test_df[target]
    logger.info(msg="Target extracted")

    return (train_X, train_y, test_X, test_y)


class Transformer(BaseEstimator, TransformerMixin):

    def __init__(self, features: FeatureParams):
        self.cat_features = features.categorical_features
        self.num_features = features.numerical_features
        self.target = features.target

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame, target=None):
        self.encoder.fit(df[self.cat_features])
        self.scaler.fit(df[self.num_features])
        logger.info(msg="Data fitted")
        return self

    def transform_categorical_features(self, df_cat: pd.DataFrame) -> np.ndarray:
        return self.encoder.transform(df_cat)

    def transform_numeric_features(self, df: pd.DataFrame) -> np.ndarray:
        transformed_num = self.scaler.fit_transform(df.to_numpy())
        return transformed_num

    def transform(self, df: pd.DataFrame, target=None) -> pd.DataFrame:
        np_transformed_cat = self.transform_categorical_features(df[self.cat_features])
        np_transformed_num = self.transform_numeric_features(df[self.num_features])
        logger.info(msg="Data transformed")
        return np.concatenate((np_transformed_cat, np_transformed_num), axis=1)
