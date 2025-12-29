# api/index.py
from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import requests
import tempfile
import zipfile
import io

app = Flask(__name__)

# Fungsi untuk mengunduh model dari GitHub
def download_models():
    try:
        # URL raw dari file model
        repo_owner = "KazeYuuji"
        repo_name = "ML4"
        branch = "main"
        
        models = {
            'camera_price_prediction_model.pkl': f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/models/camera_price_prediction_model.pkl",
            'camera_price_scaler.pkl': f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/models/camera_price_scaler.pkl",
            'camera_brand_encoder.pkl': f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/models/camera_brand_encoder.pkl",
            'selected_features.pkl': f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/models/selected_features.pkl",
            'feature_importance.csv': f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/models/feature_importance.csv"
        }
        
        # Buat direktori models
        os.makedirs('models', exist_ok=True)
        
        # Download setiap model
        for filename, url in models.items():
            response = requests.get(url)
            response.raise_for_status()
            with open(f'models/{filename}', 'wb') as f:
                f.write(response.content)
        
        return True
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False

# Download models saat startup
models_downloaded = download_models()

# Load model dan objek preprocessing
try:
    if models_downloaded:
        model = joblib.load('models/camera_price_prediction_model.pkl')
        scaler = joblib.load('models/camera_price_scaler.pkl')
        le = joblib.load('models/camera_brand_encoder.pkl')
        selected_features = joblib.load('models/selected_features.pkl')
        feature_importance = pd.read_csv('models/feature_importance.csv')
        model_loaded = True
        print("Models loaded successfully!")
    else:
        model_loaded = False
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Tahun saat ini untuk fitur usia kamera
current_year = datetime.now().year

# --- ROUTES ---

@app.route('/')
def home():
    return send_from_directory('public', 'index.html')

@app.route('/about')
def about():
    return send_from_directory('public', 'about.html')

@app.route('/api/feature-importance')
def feature_importance_api():
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        top_features = feature_importance.head(10).to_dict('records')
        return jsonify({
            'success': True,
            'features': top_features
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        # Ambil data dari form
        camera_specs = {
            'Model': request.form['model'],
            'Release date': int(request.form['release_date']),
            'Max resolution': int(request.form['max_resolution']),
            'Low resolution': int(request.form['low_resolution']),
            'Effective pixels': int(request.form['effective_pixels']),
            'Zoom wide (W)': float(request.form['zoom_wide']),
            'Zoom tele (T)': float(request.form['zoom_tele']),
            'Normal focus range': float(request.form['normal_focus']),
            'Macro focus range': float(request.form['macro_focus']),
            'Storage included': int(request.form['storage']),
            'Weight (inc. batteries)': float(request.form['weight']),
            'Dimensions': float(request.form['dimensions'])
        }
        
        # Terapkan feature engineering
        new_df = pd.DataFrame([camera_specs])
        new_df['Brand'] = new_df['Model'].apply(lambda x: x.split()[0])
        new_df['Model_Name'] = new_df['Model'].apply(lambda x: ' '.join(x.split()[1:]))
        new_df['Camera_Age'] = current_year - new_df['Release date']
        new_df['Zoom_Ratio'] = new_df['Zoom tele (T)'] / new_df['Zoom wide (W)']
        new_df['Zoom_Ratio'] = new_df['Zoom_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
        new_df['Resolution_Ratio'] = new_df['Max resolution'] / new_df['Low resolution']
        new_df['Resolution_Ratio'] = new_df['Resolution_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(1)
        new_df['Megapixels'] = new_df['Effective pixels'] / 1000000
        new_df['Weight_per_MP'] = new_df['Weight (inc. batteries)'] / new_df['Megapixels']
        new_df['Weight_per_MP'] = new_df['Weight_per_MP'].replace([np.inf, -np.inf], np.nan).fillna(185)
        new_df['Size_Efficiency'] = new_df['Max resolution'] / new_df['Dimensions']
        new_df['Size_Efficiency'] = new_df['Size_Efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Encode brand
        try:
            new_df['Brand_Encoded'] = le.transform(new_df['Brand'])
        except ValueError:
            new_df['Brand_Encoded'] = 0
        
        # Buat brand dummies
        brand_dummies_new = pd.get_dummies(new_df['Brand'], prefix='Brand', drop_first=True)
        new_df = pd.concat([new_df, brand_dummies_new], axis=1)
        
        # Pastikan semua kolom yang diperlukan ada
        for col in selected_features:
            if col not in new_df.columns:
                new_df[col] = 0
        
        # Pilih hanya fitur yang digunakan dalam model
        new_df_selected = new_df[selected_features]
        
        # Skalakan fitur
        new_df_scaled = scaler.transform(new_df_selected)
        
        # Buat prediksi
        prediction = model.predict(new_df_scaled)[0]
        
        # Dapatkan interval prediksi
        tree_predictions = np.array([tree.predict(new_df_scaled)[0] for tree in model.estimators_])
        std_dev = np.std(tree_predictions)
        confidence_interval = (prediction - 1.96*std_dev, prediction + 1.96*std_dev)
        
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'confidence_interval': (
                round(confidence_interval[0], 2),
                round(confidence_interval[1], 2)
            ),
            'brand': new_df['Brand'].values[0]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
