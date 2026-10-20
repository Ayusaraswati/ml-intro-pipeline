import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def create_sample_dataset():
    """
    Membuat sampel data historis perumahan (Data-Driven Input).
    Fitur: Ukuran (sqft), Jumlah Kamar Tidur, Lokasi (Kode Wilayah 1 atau 2).
    Target: Harga Rumah (dalam ribuan USD).
    """
    data = {
        'size_sqft': [850, 1200, 1500, 1800, 2100, 2400, 950, 1350, 1600, 2200],
        'bedrooms': [2, 2, 3, 3, 4, 4, 2, 3, 3, 4],
        'location_code': [1, 1, 2, 2, 2, 1, 1, 2, 1, 2], # 1: Pinggiran Kota, 2: Pusat Kota
        'price': [150, 210, 260, 320, 390, 410, 170, 240, 280, 400]
    }
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/housing_sample.csv", index=False)
    print("[*] Dataset sampel berhasil dibuat di 'data/housing_sample.csv'.")
    return df

def run_ml_pipeline():
    print("=== [ML Pipeline] Memulai Proses Data-Driven Training ===")
    
    # 1. Load Data
    if not os.path.exists("data/housing_sample.csv"):
        df = create_sample_dataset()
    else:
        df = pd.read_csv("data/housing_sample.csv")
    
    print(f"[*] Total record data historis: {len(df)} baris.")
    
    # 2. Definisikan Input Features (X) dan Target (y)
    X = df[['size_sqft', 'bedrooms', 'location_code']]
    y = df['price']
    
    # 3. Split Data: Training (80%) dan Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Data training: {len(X_train)} baris | Data testing: {len(X_test)} baris.")
    
    # 4. Pilih & Latih Model (Supervised Learning - Linear Regression)
    model = LinearRegression()
    print("[*] Melatih model (Model Training & Fitting)...")
    model.fit(X_train, y_train)
    
    # 5. Evaluasi Model pada Data Testing (Unseen Data)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Hasil Evaluasi Performa Model ---")
    print(f"Mean Squared Error (MSE) : {mse:.4f}")
    print(f"R-squared (R2 Score)     : {r2:.4f}")
    
    # Simpan hasil metrik ke folder outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/model_metrics.txt", "w") as f:
        f.write(f"Model Evaluation Report\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"R2 Score: {r2}\n")
    print("[✓] Laporan evaluasi berhasil disimpan ke 'outputs/model_metrics.txt'.")
    
    # 6. Simulasi Prediksi Data Baru (Inference)
    print("\n--- Simulasi Inference (Prediksi Rumah Baru) ---")
    # Kasus: Rumah baru dengan ukuran 1700 sqft, 3 kamar tidur, berlokasi di Pusat Kota (kode 2)
    new_house = [[1700, 3, 2]]
    predicted_price = model.predict(new_house)
    print(f"Input Fitur [Size: 1700 sqft, Bed: 3, Location: 2]")
    print(f"[✓] Prediksi Estimasi Harga Rumah: ${predicted_price[0]:.2f} ribu USD")

if __name__ == "__main__":
    run_ml_pipeline()
