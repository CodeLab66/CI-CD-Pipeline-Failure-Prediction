"""Interactive Flask frontend for the CI/CD failure prediction model."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from app.model_service import load_dashboard_data, predict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    CORS(app)

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/health')
    def health():
        data = load_dashboard_data()
        return jsonify({'status': 'ok', 'model_available': data['model_available']})

    @app.get('/api/model-info')
    def model_info():
        return jsonify(load_dashboard_data())

    @app.post('/api/predict')
    def predict_route():
        payload = request.get_json(silent=True) or {}
        return jsonify(predict(payload))

    @app.get('/models/<path:filename>')
    def model_artifact(filename):
        return send_from_directory(PROJECT_ROOT / 'models', filename)

    return app


app = create_app()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
