import unittest

from fastapi.testclient import TestClient
from online_inference.app import app


class TestOnlineInference(unittest.TestCase):

    def setUp(self):
        self.data_to_predict = {
            "features_names": [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
            ],
            "data": [
                [50, 1, 0, 150, 222, 1, 2, 111, 0, 0.1, 1, 1, 0],
                [50, 0, 0, 130, 239, 0, 0, 155, 0, 1.5, 0, 2, 0],
                [90, 0, 0, 130, 239, 0, 0, 155, 0, 1.5, 0, 2, 0],
            ],
            "model": "logreg",
        }

    def test_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), "predictor is alive :)")

    def test_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_predict_empty_data(self):
        response = client.get("/predict")
        self.assertEqual(response.status_code, 405)

    def test_predict_error_data(self):
        response = client.post("/predict", json={})
        self.assertEqual(response.status_code, 400) # Не прошли валидацию

        error_data = self.data_to_predict
        error_data["model"] = "randon model"
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["data"] = [[627, 1, 0, 150, "200", 1, 2, 131, 0, 0.1, 1, 1, 0],
                              [27, 1, 0, 150, 200, 1, 2, 131, 0, 0.1, 1, 1, 0]]
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["data"] = [[1]]
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

        error_data = self.data_to_predict
        error_data["features_names"] = ["hello", "world"]
        response = client.post("/predict", json=error_data)
        self.assertEqual(response.status_code, 400)

    def test_predict_ok(self):
        response = client.post("/predict", json=self.data_to_predict)
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    client = TestClient(app)
    unittest.main()
