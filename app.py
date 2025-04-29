import os
from flask import (
    Flask, render_template, request,
    send_from_directory, jsonify, flash
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import config

app = Flask(__name__)
app.config.from_object(config)
app.secret_key = 'your-secret-key'

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# ① Color adjustment endpoint
@app.route('/adjust', methods=['POST'])
def adjust():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'invalid file'}), 400

    # Save original upload to UPLOAD_FOLDER
    original_fn = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_fn)
    file.save(upload_path)

    # HSV parameters from form
    hue_shift = float(request.form.get('hue', 0))
    sat_mul   = float(request.form.get('sat', 100)) / 100.0
    val_mul   = float(request.form.get('val', 100)) / 100.0

    # Read image and convert to HSV
    img = cv2.imread(upload_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Apply shifts and multipliers
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mul, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_mul, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Save adjusted image to PROCESSED_FOLDER
    adjusted_fn = f"adjusted_{original_fn}"
    adjusted_path = os.path.join(app.config['PROCESSED_FOLDER'], adjusted_fn)
    cv2.imwrite(adjusted_path, adjusted)

    return jsonify({'filename': adjusted_fn})

# ② Color quantization endpoint
@app.route('/quantize', methods=['POST'])
def quantize():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'no filename provided'}), 400

    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(processed_path):
        return jsonify({'error': 'file not found'}), 404

    img = cv2.imread(processed_path)
    if img is None:
        return jsonify({'error': 'cannot read image'}), 400

    # Optionally resize/compress
    resize_and_compress_opencv(
        processed_path, processed_path,
        max_width=800, max_height=800, target_kb=250
    )

    # Perform k-means quantization
    K = int(data.get('num_clusters', config.NUM_CLUSTERS))
    small = cv2.resize(
        img, (0, 0),
        fx=config.RESIZE_SCALE,
        fy=config.RESIZE_SCALE
    )
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.TERM_CRITERIA_MAX_ITER,
        config.TERM_CRITERIA_EPS
    )
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria,
        config.KMEANS_ATTEMPTS,
        cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(small.shape)

    quant_fn = f"quantized_{filename}"
    quant_path = os.path.join(app.config['PROCESSED_FOLDER'], quant_fn)
    cv2.imwrite(quant_path, quantized)

    return jsonify({'filename': quant_fn})

@app.route('/processed/<filename>')
def processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Utility function: resize and compress

def resize_and_compress_opencv(input_path, output_path, max_width, max_height, target_kb):
    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w*scale), int(h*scale)),
            interpolation=cv2.INTER_AREA
        )
    max_bytes = target_kb * 1024
    quality = 95
    while True:
        ok, buf = cv2.imencode(
            '.jpg', img,
            [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok or len(buf) <= max_bytes or quality <= 50:
            break
        quality -= 5
    with open(output_path, 'wb') as f:
        f.write(buf)

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )
