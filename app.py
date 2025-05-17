import os
from flask import (
    Flask, render_template, request,
    send_from_directory, jsonify
)
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import config

<<<<<<< HEAD
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
=======
app = Flask(__name__)
app.config.from_object(config)
app.secret_key = 'your-secret-key'

# フォルダ準備
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ---------------------------------
# ① トップページ表示
# ---------------------------------
@app.route('/', methods=['GET'])
def index():
    # index.htmlテンプレートをレンダリング
    return render_template('index.html')

<<<<<<< HEAD
# ---------------------------------
# ② 色調整機能
#    色相(hue)、彩度(sat)、明度(val)をスライダーで調整
# ---------------------------------
=======
# 色調整
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
@app.route('/adjust', methods=['POST'])
def adjust():
    file = request.files.get('image')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'invalid file'}), 400

    fn = secure_filename(file.filename)
    in_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    file.save(in_path)

<<<<<<< HEAD
    # パラメータ取得
=======
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
    hue = float(request.form.get('hue', 0))
    sat = float(request.form.get('sat', 100)) / 100.0
    val = float(request.form.get('val', 100)) / 100.0

    img = cv2.imread(in_path)
<<<<<<< HEAD
    # HSV空間で調整
=======
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + hue) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sat, 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * val, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    out_fn = f"adjusted_{fn}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, out)
    return jsonify({'filename': out_fn})

<<<<<<< HEAD
# ---------------------------------
# ③ ぼかし機能
#    GaussianBlurで画像をぼかす
# ---------------------------------
=======
# ぼかし
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
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

<<<<<<< HEAD
# ---------------------------------
# ④ ポスタライズ機能
#    指定階調数levelsで色数を制限
# ---------------------------------
=======
# ポスタライズ
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
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
<<<<<<< HEAD
    step = 255.0 / (levels - 1)
=======
    # Improved uniform posterization: nearest of equally spaced levels
    step = 255.0 / (levels - 1)
    # Apply per-channel mapping
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
    poster = np.round(img.astype(np.float32) / step) * step
    poster = np.clip(poster, 0, 255).astype(np.uint8)

    out_fn = f"posterize_{levels}_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, poster)
    return jsonify({'filename': out_fn})

<<<<<<< HEAD
# ---------------------------------
# ⑤ 境界線抽出機能
#    Cannyエッジ検出＋反転で線画化
# ---------------------------------
@app.route('/edges', methods=['POST'])
def edges():
    filename = request.form.get('filename')
    try:
=======


@app.route('/edges', methods=['POST'])
def edges():
    """⑤ 境界線抽出──Canny→反転で細い黒線のみを描画"""
    filename = request.form.get('filename')
    try:
        # optional: スライダーで調整したい場合はフォームから threshold を取る
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
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
<<<<<<< HEAD

    edges = cv2.Canny(img, thresh1, thresh2)
    inv = cv2.bitwise_not(edges)
    out = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    out_fn = f"edges_{thresh1}_{thresh2}_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, out)
    return jsonify({'filename': out_fn})
=======

    # Canny でエッジ検出 → 白線 on 黒背景 → invert → 黒線 on 白背景
    edges = cv2.Canny(img, thresh1, thresh2)
    inv = cv2.bitwise_not(edges)
    # convert single→3ch so browser can display as jpg/png
    out = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    out_fn = f"edges_{thresh1}_{thresh2}_{filename}"
    out_path = os.path.join(app.config['PROCESSED_FOLDER'], out_fn)
    cv2.imwrite(out_path, out)
    return jsonify({'filename': out_fn})




>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46

# ---------------------------------
# ⑥ プロセス済みファイル配信エンドポイント
# ---------------------------------
@app.route('/processed/<filename>')
def processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

<<<<<<< HEAD

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
=======
def resize_and_compress_opencv(input_path, output_path, max_width, max_height, target_kb):
    img = cv2.imread(input_path)
    h, w = img.shape[:2]
    scale = min(max_width/w, max_height/h)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    max_bytes = target_kb * 1024
    q = 95
    while True:
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok or len(buf) <= max_bytes or q <= 50:
            break
        q -= 5
    with open(output_path, 'wb') as f:
        f.write(buf)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
>>>>>>> 69f2387b42e400c360afba983bd9f023c22b0f46
