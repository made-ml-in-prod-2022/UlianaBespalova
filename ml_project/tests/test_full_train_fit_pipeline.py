import json
import pickle
import unittest

from train_test_pipeline import train_model_pipeline


class TestFullPipeline(unittest.TestCase):

    def test_train_pipeline(self):
        config_path = 'tests/test_config_fit.yaml'
        train_model_pipeline(config_path)

        with open('metrics.json', "w+") as file:
            metrics = json.load(file)
        self.assertTrue(metrics["mae"] < 0.4)

        with open('model.pkl', "wb") as file:
            model = pickle.load(file)
        self.assertTrue(model, "RandomForestClassifier(max_depth=10, n_estimators=500, random_state=42")


if __name__ == '__main__':
    unittest.main()
