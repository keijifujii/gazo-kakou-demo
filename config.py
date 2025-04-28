# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'static', 'processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
# k-means パラメータ
NUM_CLUSTERS = 20
TERM_CRITERIA_MAX_ITER = 10
TERM_CRITERIA_EPS = 1.0
KMEANS_ATTEMPTS = 10
RESIZE_SCALE = 0.5
