import unittest

from enities import FeatureParams
from features import Transformer, extract_target
import pandas as pd


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.features_params = FeatureParams(
            categorical_features='Type',
            numerical_features='Age',
            target='target'
        )
        self.transformer = Transformer(self.features_params)

        data = [['A', 10], ['A', 15], ['B', 100]]
        self.df = pd.DataFrame(data, columns=['Type', 'Age'])

    def test_fit_transformer(self):
        self.transformer.fit()
        self.assertIsNotNone(self.transformer.scaler)
        self.assertIsNotNone(self.transformer.encoder)

    def test_transform_categorical_features(self):
        trsf_features = self.transformer.transform_categorical_features(
            self.df[self.features_params.categorical_features])
        self.assertEqual(list(trsf_features), [0, 0, 1])

    def test_transform_numeric_features(self):
        trsf_features = self.transformer.transform_categorical_features(
            self.df[self.features_params.numerical_features])
        self.assertEqual(list(trsf_features), [0.1, 0.15, 1])

    def test_transform(self):
        trsf_df = self.transformer.transform(self.df)
        self.assertEqual(trsf_df, [[0, 0.1], [0, 0.15], [1, 1]])


if __name__ == '__main__':
    unittest.main()
