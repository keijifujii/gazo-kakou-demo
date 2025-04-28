# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import config

# アプリ＆設定読み込み
app = Flask(__name__)
app.config.from_object(config)
app.secret_key = 'your-secret-key'  # 必要に応じて変更

# 必要フォルダがなければ作成（ここで最初にやる）
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# --- 以下、ルート定義 ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        flash('ファイルが選択されていません。')
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        flash('有効な画像ファイルを選択してください。')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    try:
        K = int(request.form.get('num_clusters', app.config['NUM_CLUSTERS']))
    except ValueError:
        K = app.config['NUM_CLUSTERS']

    img = cv2.imread(upload_path)
    if img is None:
        flash('画像の読み込みに失敗しました。')
        return redirect(url_for('index'))
        
    img_small = cv2.resize(img, (0, 0), fx=app.config.get('RESIZE_SCALE', 0.5),
                                 fy=app.config.get('RESIZE_SCALE', 0.5))

    data = img_small.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                app.config.get('TERM_CRITERIA_MAX_ITER', 10),
                app.config.get('TERM_CRITERIA_EPS', 1.0))
    attempts = app.config.get('KMEANS_ATTEMPTS', 10)
    flags = cv2.KMEANS_PP_CENTERS

    _, labels, centers = cv2.kmeans(data, K, None, criteria, attempts, flags)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized_img = quantized.reshape(img_small.shape)

    out_filename = f"quantized_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_filename)
    cv2.imwrite(out_path, quantized_img)

    return render_template('result.html', filename=out_filename, original=filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/original/<filename>')
def original_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- 起動設定（これが一番最後） ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
