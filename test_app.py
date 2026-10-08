
import unittest
from app import app


class TestHeartApp(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["status"], "healthy"
        )

    def test_prediction(self):
        data = {
            "age": 52,
            "sex": 1,
            "cp": 0,
            "trestbps": 125,
            "chol": 212,
            "fbs": 0,
            "restecg": 1,
            "thalach": 168,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 2,
            "ca": 2,
            "thal": 3
        }

        response = self.client.post(
            "/predict", json=data
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())


if __name__ == "__main__":
    unittest.main()
