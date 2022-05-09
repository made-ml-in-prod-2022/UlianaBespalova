import json
import pickle
import unittest

from sklearn.linear_model import LogisticRegression

from enities import ModelParams, LogRegParams, ForestParams, get_train_params, get_predict_params
from models import (
    test_model,
    get_model,
    save_metrics_to_json,
    serialize_model,
    open_model,
    save_predict,
)


class TestFitPredictModel(unittest.TestCase):

    def setUp(self):
        logreg_params = LogRegParams(max_iter=800, random_state=42)
        forest_params = ForestParams(n_estimators=500, max_depth=10, random_state=42)
        self.model_params = ModelParams(save_path="models/model.pkl",
                                        metric_path="models/metrics.json",
                                        model="RandomForest",
                                        logreg_params=logreg_params,
                                        forest_params=forest_params
                                        )

    def test_get_model(self):
        model = get_model(self.model_params)
        self.assertEqual(model, "RandomForestClassifier(max_depth=10, n_estimators=500, random_state=42")

    def test_save_metrics_to_json(self):
        metrics_to_save = {'m1': 1, 'm2': 2}
        save_path = '.'
        save_metrics_to_json(metrics_to_save, save_path)
        with open(save_path, "w+") as file:
            metrics = json.load(file)
        self.assertEqual(metrics_to_save, metrics)

    def test_serialize_model(self):
        model_to_save = LogisticRegression()
        path = '.'
        serialize_model(model_to_save, path)
        with open(path, "wb") as f:
            model = pickle.load(f)
        self.assertEqual(model, model_to_save)

    def test_open_model(self):
        model_to_save = LogisticRegression()
        path = '.'
        with open(path, "wb") as f:
            pickle.dump(model_to_save, f)
        model = open_model(path)
        self.assertEqual(model, model_to_save)


if __name__ == '__main__':
    unittest.main()

# <class 'sklearn.pipeline.Pipeline'>
