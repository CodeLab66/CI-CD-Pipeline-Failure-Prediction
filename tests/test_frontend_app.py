"""Tests for the Flask frontend and prediction API."""

import unittest

from app import create_app


class FrontendAppTest(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health_reports_model_available(self):
        response = self.client.get('/api/health')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'ok')
        self.assertTrue(payload['model_available'])

    def test_model_info_contains_metrics_and_fields(self):
        response = self.client.get('/api/model-info')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn('test_metrics', payload)
        self.assertIn('accuracy', payload['test_metrics'])
        self.assertGreater(len(payload['fields']), 10)
        self.assertGreater(len(payload['feature_importance']), 0)

    def test_prediction_endpoint_returns_probability_trace(self):
        response = self.client.post('/api/predict', json={'gh_lang': 'python', 'git_diff_src_churn': 120})

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn(payload['prediction'], ['passed', 'failed_or_errored'])
        self.assertGreaterEqual(payload['failure_probability'], 0)
        self.assertLessEqual(payload['failure_probability'], 1)
        self.assertGreaterEqual(len(payload['trace']), 5)

    def test_home_page_loads_frontend(self):
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'CI/CD Failure Prediction', response.data)

    def test_saved_plot_artifact_is_served(self):
        response = self.client.get('/models/raw_roc_curve.png')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/png')
        response.close()


if __name__ == '__main__':
    unittest.main()
