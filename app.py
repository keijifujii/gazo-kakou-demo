#2025-06-02更新
import os
from flask import (
    Flask, render_template, request,
    send_from_directory, jsonify, url_for
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
import config

app = Flask(__name__)
app.config.from_object(config)
app.secret_key = 'your-secret-key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXT = set(config.ALLOWED_EXTENSIONS) | {'webp'}
def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def load_image(path):
    """WebP/JPG/PNG/GIF → NumPy BGR配列で返す"""
    with Image.open(path) as im:
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

def save_jpeg(np_bgr, out_path, quality=95):
    """NumPy BGR配列→JPGで保存"""
    im = Image.fromarray(cv2.cvtColor(np_bgr, cv2.COLOR_BGR2RGB))
    im.save(out_path, format="JPEG", quality=quality)

def to_jpg_filename(prefix, orig_filename):
    base, _ = os.path.splitext(orig_filename)
    return f"{prefix}_{base}.jpg"

# 色調整
@app.route('/adjust', methods=['POST'])
def adjust():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'invalid file'}), 400

    fn = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    file.save(in_path)

    hue = float(request.form.get('hue', 0))
    sat = float(request.form.get('sat', 100)) / 100.0
    val = float(request.form.get('val', 100)) / 100.0

    img = load_image(in_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    out_fn = to_jpg_filename("adjusted", fn)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    save_jpeg(out, out_path)

    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})

# コントラスト調整
@app.route('/contrast', methods=['POST'])
def contrast():
    file = request.files.get('image')
    if file and allowed_file(file.filename):
        fn_src = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], fn_src)
        file.save(temp_path)
        in_path = temp_path
        base_name = fn_src
    else:
        base_name = request.form.get('filename')
        in_path = os.path.join(app.config['PROCESSED_FOLDER'], base_name)
        if not os.path.exists(in_path):
            return jsonify({'error': 'file not found'}), 404

    try:
        alpha = float(request.form.get('contrast', 100)) / 100.0
    except ValueError:
        return jsonify({'error': 'invalid contrast'}), 400

    img = load_image(in_path)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    out_fn = to_jpg_filename(f"contrast_{int(alpha*100)}", base_name)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    save_jpeg(adjusted, out_path)

    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})

# ぼかし
@app.route('/blur', methods=['POST'])
def blur():
    filename = request.form.get('filename')
    try:
        radius = int(request.form.get('radius', 0))
    except ValueError:
        return jsonify({'error': 'invalid radius'}), 400

    in_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(in_path):
        return jsonify({'error': 'file not found'}), 404

    img = load_image(in_path)
    k = max(1, radius * 2 + 1)
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    out_fn = to_jpg_filename(f"blur_{radius}", filename)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    save_jpeg(blurred, out_path)
    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})

# ポスタライズ
@app.route('/posterize', methods=['POST'])
def posterize():
    filename = request.form.get('filename')
    try:
        levels = int(request.form.get('levels', 4))
        if levels < 2:
            raise ValueError
    except ValueError:
        return jsonify({'error': 'invalid levels'}), 400

    in_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(in_path):
        return jsonify({'error': 'file not found'}), 404

    img = load_image(in_path)
    step = 255.0 / (levels - 1)
    poster = np.round(img.astype(np.float32) / step) * step
    poster = np.clip(poster, 0, 255).astype(np.uint8)

    out_fn = to_jpg_filename(f"posterize_{levels}", filename)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    save_jpeg(poster, out_path)
    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})

# 境界線抽出
@app.route('/edges', methods=['POST'])
def edges():
    filename = request.form.get('filename')
    try:
        thresh1 = int(request.form.get('th1', 50))
        thresh2 = int(request.form.get('th2', 150))
    except ValueError:
        return jsonify({'error': 'invalid thresholds'}), 400

    in_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(in_path):
        return jsonify({'error': 'file not found'}), 404

    img = load_image(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, thresh1, thresh2)
    inv = cv2.bitwise_not(edges)
    out = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    out_fn = to_jpg_filename(f"edges_{thresh1}_{thresh2}", filename)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    save_jpeg(out, out_path)
    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})

# 鮮明化
@app.route('/superres', methods=['POST'])
def superres():
    filename = request.form.get('filename')
    in_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(in_path):
        return jsonify({'error': 'file not found'}), 404

    model_path = 'models/realesr-general-x4v3.pth'
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model_name='realesr-general-x4v3',
        tile=128,
        tile_pad=10,
        pre_pad=0,
        half=True
    )

    img = Image.open(in_path).convert('RGB')
    np_img = np.array(img)
    try:
        output, _ = upsampler.enhance(np_img)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    out_fn = to_jpg_filename("superres", filename)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    Image.fromarray(output).save(out_path, format="JPEG", quality=95)
    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})




@app.route('/upsample', methods=['POST'])
def upsample():
    filename = request.form.get('filename')
    try:
        scale = int(request.form.get('scale', 2))
        if scale < 1: raise ValueError
    except ValueError:
        return jsonify({'error': 'invalid scale'}), 400

    in_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(in_path):
        return jsonify({'error': 'file not found'}), 404

    img = load_image(in_path)
    h, w = img.shape[:2]
    new_w, new_h = w * scale, h * scale
    up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    out_fn = to_jpg_filename(f"upsample_{scale}x", filename)
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    save_jpeg(up, out_path)

    return jsonify({'filename': out_fn, 'url': url_for('processed', filename=out_fn)})





@app.route('/processed/<filename>')
def processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
