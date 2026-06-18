from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model = load_model("model_sampah.h5")

# HARUS SESUAI URUTAN class_indices SAAT TRAINING
labels = [
    "Botol Plastik",
    "Kaca",
    "Kaleng",
    "Kardus",
    "Kertas",
    "Plastik sampah",
    "Styrofoam"
]

EMISI = {
    "Botol Plastik": 0.35,
    "Kaca": 0.40,
    "Kaleng": 0.50,
    "Kardus": 0.18,
    "Kertas": 0.15,
    "Plastik sampah": 0.20,
    "Styrofoam": 0.30
}

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files['image']

    filepath = "temp.jpg"
    file.save(filepath)

    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x)
    print("\n=== HASIL RAW ===")
    print(pred)

    idx = np.argmax(pred)

    nama = labels[idx]

    confidence = float(np.max(pred)) * 100

    os.remove(filepath)

    print("\n===== HASIL PREDIKSI =====")
    print("Nama Produk :", nama)
    print("Confidence  :", round(confidence, 2))
    print("Emisi       :", EMISI[nama])
    print("=========================\n")

    return jsonify({
        "NamaProduk": nama,
        "Confidence": round(confidence, 2),
        "EmisiKarbon": EMISI[nama]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)