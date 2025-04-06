from flask import Flask, jsonify, request
from flask_cors import CORS
from libs.generate import generate_text
from libs.detect import detect_watermark

app = Flask(__name__)
CORS(app) 

@app.route('/generate', methods=['POST'])
def generate_handler():
    """Endpoint for text generation with watermarking"""
    try:
        data = request.get_json()
        result = generate_text(data)
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/detect', methods=['POST'])
def detect_handler():
    """Endpoint for watermark detection"""
    try:
        data = request.get_json()
        detection_result = detect_watermark(data)
        return jsonify({
            'status': 'success',
            'data': detection_result
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)