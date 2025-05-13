from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# === MAPPING KATEGORI === #
durasi_olahraga_map = {
    'Kurang dari 30 menit': 0,
    'di atas 90 menit': 1,
    '60 - 90 menit': 2,
    '30 - 60 menit': 3
}

makan_map = {
    '1 kali': 0,
    '2 kali': 1,
    'lebih dari 3 kali': 2,
    '3 kali': 3
}

tidur_map = {
    'Kurang dari 5 jam': 0,
    '6 jam': 1,
    '7 jam': 2,
    '7 - 8 jam': 3,
    '8 jam': 3,
    'Lebih dari 8 jam': 4
}

gender_map = {
    'Perempuan': 0,
    'Laki-laki': 1
}

olahraga_map = {
    "Lari": 20,
    "Jogging": 8,
    "Push up": 10,
    "Scout jump": 14,
    "Sit up": 6,
    "Senam": 9,
    "jump It": 13,
    "Sepak bola": 19,
    "Joging": 8,
    "Renang": 18,
    "jalan jalan": 0,
    "jalan pagi": 0,
    "skotjump": 14,
    "main bola": 19,
    "badminton": 11,
    "jalan-jalan": 0,
    "jalan kaki": 0,
    "Sepak bola dan jalan pagi": 19,
    "Lari pagi": 8,
    "sepeda": 12,
    "Bersepeda": 12,
    "bola": 19,
    "tenis meja": 3,
    "Berlari dan berenang": 20,
    "Bermain sepak bola": 19,
    "Berkuda": 20,
    "Rollerblade, Swimming, Dancing": 15,
    "Berenang": 18,
    "Wall Climbing": 17,
    "Roller blade": 15,
    "Jalan cepat": 0,
    "Voli": 4,
    "Lompat tali": 13,
    "jogging, renang": 8,
    "Karate": 7,
    "Basket, jogging": 16,
    "Lari kecil": 20,
    "Bulu tangkis": 11,
    "Taekwondo": 7,
    "Basket": 16,
    "Sepak bola, voli": 19,
    "Bola": 4,
    "Running": 20,
    "Futsal": 20,
    "berenang": 18,
    "bela diri (karate)": 7,
    "Jalan Santai": 0,
    "Bulu Tangkis": 11,
    "Push Up dan Lari": 10,
    "Bersepeda , Sepak Bola": 12,
    "Sepak Bola , Lari dan Bersepeda": 19,
    "Sepak Bola , Berenang dan Bersepeda": 19,
    "Bersepeda dan Lari": 12,
    "Berenang , Bersepeda": 18,
    "Bermain Lompat Tali": 13,
    "Tari": 5,
    "Sepeda": 12,
    "Bola Voli": 4,
    "Bersepa": 12,
    "Jogging": 20,
    "berlari": 20,
    "Balet": 5,
    "Aerobik": 9,
    "Lari/jogging": 20,
    "Lompat Indah": 13,
    "Bermain hula hoop": 1,
    "Bermain bulu tangkis": 11,
    "Yoga": 2,
    "skipping": 13,
    "Bermain Voli": 4,
    "Pilates": 2,
    "ballet": 5,
    "bersepeda, bola": 12,
    "Bulutangkis": 11
}

required_fields = [
    'Usia',
    'Jenis Kelamin',
    'Berat Badan',
    'Tinggi Badan',
    'Lingkar Pinggang (M)',
    'Makan Dalam Sehari',
    'Durasi Olahraga',
    'Jenis Olahraga',
    'Durasi Tidur'
]

# Nama kelas berdasarkan label numerik
class_names = {
    0: 'Berat badan berlebih',
    1: 'Kurang Gizi',
    2: 'Normal',
    3: 'Obesitas'
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f'Field wajib belum lengkap: {missing}'}), 400

    try:
        usia = float(data['Usia'])
        gender = gender_map[data['Jenis Kelamin']]
        berat = float(data['Berat Badan'])
        tinggi = float(data['Tinggi Badan'])
        lingkar = float(data['Lingkar Pinggang (M)'])
        makan = makan_map[data['Makan Dalam Sehari']]
        durasi_olahraga = durasi_olahraga_map[data['Durasi Olahraga']]
        jenis_olahraga = olahraga_map[data['Jenis Olahraga'].strip()]
        tidur = tidur_map[data['Durasi Tidur']]
    except KeyError as e:
        return jsonify({'error': f'Nilai tidak dikenali: {e}'}), 400
    except ValueError:
        return jsonify({'error': 'Input numerik tidak valid'}), 400

    # Buat array input
    features = [usia, gender, berat, tinggi, lingkar, makan, durasi_olahraga, jenis_olahraga, tidur]
    input_array = np.array(features).reshape(1, -1)

    # Mengambil probabilitas untuk setiap kelas
    probabilities = model.predict_proba(input_array)[0]  # [0] untuk mengambil prediksi untuk satu instance

    # Menentukan kelas prediksi
    predicted_class = class_names[np.argmax(probabilities)]

    # Menampilkan probabilitas kelas yang diprediksi
    class_probabilities = {class_names[i]: prob for i, prob in enumerate(probabilities)}


    bmi = berat / (tinggi / 100) ** 2  # tinggi diubah ke meter


    return jsonify({
        'input': data,
        'prediction': predicted_class,
        'probabilities': class_probabilities , # Mengembalikan probabilitas untuk setiap kelas
        'bmi' : bmi
    })

@app.route('/')
def index():
    return "BMI Decision Tree API is running."

def handler(environ, start_response):
    return app(environ, start_response)
