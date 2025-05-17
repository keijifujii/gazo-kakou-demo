import os
from flask import (
    Flask, render_template, request,
    send_from_directory, jsonify
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import config

from realesrgan import RealESRGANer  # AI超解像モデルを利用
from PIL import Image

# Flaskアプリケーションの作成
app = Flask(__name__)
app.config.from_object(config)  # 設定読み込み
app.secret_key = 'your-secret-key'  # セッションなどで使用

# アップロード・処理後フォルダの準備
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# 許可する画像ファイルか判定するユーティリティ関数
def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ---------------------------------
# ① トップページ表示
# ---------------------------------
@app.route('/', methods=['GET'])
def index():
    # index.htmlテンプレートをレンダリング
    return render_template('index.html')

# ---------------------------------
# ② 色調整機能
#    色相(hue)、彩度(sat)、明度(val)をスライダーで調整
# ---------------------------------
@app.route('/adjust', methods=['POST'])
def adjust():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'invalid file'}), 400

    fn = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    file.save(in_path)

    # パラメータ取得
    hue = float(request.form.get('hue', 0))
    sat = float(request.form.get('sat', 100)) / 100.0
    val = float(request.form.get('val', 100)) / 100.0

    img = cv2.imread(in_path)
    # HSV空間で調整
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + hue) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sat, 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * val, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    out_fn = f"adjusted_{fn}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, out)
    return jsonify({'filename': out_fn})

# ---------------------------------
# ③ ぼかし機能
#    GaussianBlurで画像をぼかす
# ---------------------------------
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

    img = cv2.imread(in_path)
    k = max(1, radius * 2 + 1)
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    out_fn = f"blur_{radius}_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, blurred)
    return jsonify({'filename': out_fn})

# ---------------------------------
# ④ ポスタライズ機能
#    指定階調数levelsで色数を制限
# ---------------------------------
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

    img = cv2.imread(in_path)
    step = 255.0 / (levels - 1)
    poster = np.round(img.astype(np.float32) / step) * step
    poster = np.clip(poster, 0, 255).astype(np.uint8)

    out_fn = f"posterize_{levels}_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, poster)
    return jsonify({'filename': out_fn})

# ---------------------------------
# ⑤ 境界線抽出機能
#    Cannyエッジ検出＋反転で線画化
# ---------------------------------
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

    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({'error': 'cannot read image'}), 400

    edges = cv2.Canny(img, thresh1, thresh2)
    inv = cv2.bitwise_not(edges)
    out = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    out_fn = f"edges_{thresh1}_{thresh2}_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, out)
    return jsonify({'filename': out_fn})

# ---------------------------------
# ⑥ プロセス済みファイル配信エンドポイント
# ---------------------------------
@app.route('/processed/<filename>')
def processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


# ---------------------------------
# ⑦ 鮮明化（超解像）機能
#    Real-ESRGANを使って4倍超解像
# ---------------------------------
@app.route('/superres', methods=['POST'])
def superres():
    filename = request.form.get('filename')
    in_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(in_path):
        return jsonify({'error': 'file not found'}), 404

    # モデルロード設定
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

    out_fn = f"superres_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    Image.fromarray(output).save(out_path)
    return jsonify({'filename': out_fn})

# ---------------------------------
# アプリ起動
# ---------------------------------
if __name__ == '__main__':
    # DEBUGモードで起動（エラー時に詳細表示）
    app.run(host='0.0.0.0',
             port=int(os.environ.get('PORT', 5000)),
             debug=True)
