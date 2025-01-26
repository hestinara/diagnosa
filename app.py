from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

app = Flask(__name__)

# Data pelatihan untuk cabai dan tomat
data_cabai = [
    {
        "Daun layu": 1,
        "Tulang daun menebal": 0,
        "Daun menguning": 1,
        "Buah mengering": 1,
        "Buah bercak mengkilap": 0,
        "Batang rusak": 0,
        "Daun coklat": 0,
        "Batang menguning": 1,
        "Buah busuk": 0,
        "Buah berubah warna": 0,
        "Luka melebar": 0,
        "Daun rontok": 0,
        "Tidak berbuah": 0,
        "Akar rusak": 0,
        "Buah berair": 0,
        "Daun mengecil": 0,
        "Batang kecoklatan": 0,
        "Daun keriting": 0,
        "Buah keriput": 0,
        "Tanaman mengerdil": 0,
        "Akar coklat": 0,
        "penyakit": "Layu Fusarium (Fusarium oxysporum)"
    },
    {
        "Daun layu": 0,
        "Tulang daun menebal": 0,
        "Daun menguning": 0,
        "Buah mengering": 0,
        "Buah bercak mengkilap": 1,
        "Batang rusak": 1,
        "Daun coklat": 0,
        "Batang menguning": 0,
        "Buah busuk": 1,
        "Buah berubah warna": 0,
        "Luka melebar": 0,
        "Daun rontok": 0,
        "Tidak berbuah": 0,
        "Akar rusak": 0,
        "Buah berair": 0,
        "Daun mengecil": 0,
        "Batang kecoklatan": 0,
        "Daun keriting": 0,
        "Buah keriput": 0,
        "Tanaman mengerdil": 0,
        "Akar coklat": 0,
        "penyakit": "Layu Bakteri Ralstonia (Ralstonia solanacearum)"
    },
    {
        "Daun layu": 0,
        "Tulang daun menebal": 0,
        "Daun menguning": 1,
        "Buah mengering": 1,
        "Buah bercak mengkilap": 0,
        "Batang rusak": 1,
        "Daun coklat": 0,
        "Batang menguning": 0,
        "Buah busuk": 1,
        "Buah berubah warna": 0,
        "Luka melebar": 0,
        "Daun rontok": 0,
        "Tidak berbuah": 0,
        "Akar rusak": 0,
        "Buah berair": 0,
        "Daun mengecil": 0,
        "Batang kecoklatan": 0,
        "Daun keriting": 0,
        "Buah keriput": 0,
        "Tanaman mengerdil": 0,
        "Akar coklat": 0,
        "penyakit": "Virus Kuning (Gemini Virus)"
    },
    {
        "Daun layu": 0,
        "Tulang daun menebal": 0,
        "Daun menguning": 0,
        "Buah mengering": 0,
        "Buah bercak mengkilap": 1,
        "Batang rusak": 0,
        "Daun coklat": 0,
        "Batang menguning": 0,
        "Buah busuk": 0,
        "Buah berubah warna": 1,
        "Luka melebar": 1,
        "Daun rontok": 0,
        "Tidak berbuah": 0,
        "Akar rusak": 0,
        "Buah berair": 0,
        "Daun mengecil": 0,
        "Batang kecoklatan": 1,
        "Daun keriting": 0,
        "Buah keriput": 1,
        "Tanaman mengerdil": 0,
        "Akar coklat": 0,
        "penyakit": "Busuk Buah Antraknosa (Colletotrichum gloeosporioides)"
    },
    {
        "Daun layu": 0,
        "Tulang daun menebal": 0,
        "Daun menguning": 1,
        "Buah mengering": 0,
        "Buah bercak mengkilap": 0,
        "Batang rusak": 1,
        "Daun coklat": 1,
        "Batang menguning": 0,
        "Buah busuk": 0,
        "Buah berubah warna": 0,
        "Luka melebar": 0,
        "Daun rontok": 1,
        "Tidak berbuah": 1,
        "Akar rusak": 1,
        "Buah berair": 0,
        "Daun mengecil": 0,
        "Batang kecoklatan": 0,
        "Daun keriting": 0,
        "Buah keriput": 0,
        "Tanaman mengerdil": 0,
        "Akar coklat": 0,
        "penyakit": "Bercak Daun (Cercospora sp.)"
    }
]
data_tomat = [
    {
        "Daun bercak coklat": 1,
        "Daun tua menguning": 0,
        "Daun muda layu": 0,
        "Tangkai daun berwarna putih": 1,
        "Tangkai daun merunduk": 0,
        "Bawah daun bercak putih": 1,
        "Batang berwarna coklat": 1,
        "Batas atas mengering": 0,
        "Tanaman layu keseluruhan": 0,
        "penyakit": "Busuk Daun"
    },
    {
        "Daun bercak coklat": 0,
        "Daun tua menguning": 1,
        "Daun muda layu": 0,
        "Tangkai daun berwarna putih": 0,
        "Tangkai daun merunduk": 0,
        "Bawah daun bercak putih": 0,
        "Batang berwarna coklat": 1,
        "Batas atas mengering": 1,
        "Tanaman layu keseluruhan": 0,
        "penyakit": "Layu Fusarium"
    },
    {
        "Daun bercak coklat": 0,
        "Daun tua menguning": 0,
        "Daun muda layu": 1,
        "Tangkai daun berwarna putih": 0,
        "Tangkai daun merunduk": 1,
        "Bawah daun bercak putih": 0,
        "Batang berwarna coklat": 0,
        "Batas atas mengering": 0,
        "Tanaman layu keseluruhan": 1,
        "penyakit": "Layu Bakteri"
    },
    {
        "Daun bercak coklat": 1,
        "Daun tua menguning": 0,
        "Daun muda layu": 0,
        "Tangkai daun berwarna putih": 1,
        "Tangkai daun merunduk": 0,
        "Bawah daun bercak putih": 1,
        "Batang berwarna coklat": 0,
        "Batas atas mengering": 0,
        "Tanaman layu keseluruhan": 0,
        "penyakit": "Bercak Coklat"
    },
    {
        "Daun bercak coklat": 0,
        "Daun tua menguning": 0,
        "Daun muda layu": 1,
        "Tangkai daun berwarna putih": 0,
        "Tangkai daun merunduk": 1,
        "Bawah daun bercak putih": 0,
        "Batang berwarna coklat": 0,
        "Batas atas mengering": 0,
        "Tanaman layu keseluruhan": 1,
        "penyakit": "Hawar Daun"
    }
]

# Melatih model DecisionTreeClassifier
def train_model(data):
    df = pd.DataFrame(data)
    X = df.drop(columns=['penyakit'])
    y = df['penyakit']
    model = DecisionTreeClassifier()
    
    model.fit(X, y)
    return model

model_cabai = train_model(data_cabai)
model_tomat = train_model(data_tomat)

# Daftar gejala untuk cabai dan tomat
gejala_columns_cabai = [
    "Daun layu",
    "Tulang daun menebal",
    "Daun menguning",
    "Buah mengering",
    "Buah bercak mengkilap",
    "Batang rusak",
    "Daun coklat",
    "Batang menguning",
    "Buah busuk",
    "Buah berubah warna",
    "Luka melebar",
    "Daun rontok",
    "Tidak berbuah",
    "Akar rusak",
    "Buah berair",
    "Daun mengecil",
    "Batang kecoklatan",
    "Daun keriting",
    "Buah keriput",
    "Tanaman mengerdil",
    "Akar coklat"
]
gejala_columns_tomat = [
    "Daun bercak coklat",
    "Daun tua menguning",
    "Daun muda layu",
    "Tangkai daun berwarna putih",
    "Tangkai daun merunduk",
    "Bawah daun bercak putih",
    "Batang berwarna coklat",
    "Batas atas mengering",
    "Tanaman layu keseluruhan"
]
# Fungsi untuk mendiagnosis penyakit
def diagnosa_penyakit(gejala, model, gejala_columns):
    gejala_vector = [0] * len(gejala_columns)
    for gejala_item in gejala:
        if gejala_item in gejala_columns:
            gejala_vector[gejala_columns.index(gejala_item)] = 1
    diagnosis = model.predict([gejala_vector])
    return diagnosis[0]

# Routes untuk halaman
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cabai')
def cabai():
    return render_template('cabai.html')

@app.route('/tomat')
def tomat():
    return render_template('tomat.html')

@app.route('/gejala_cabai', methods=['GET', 'POST'])
def gejala_cabai():
    if request.method == 'POST':
        gejala = request.form.getlist('gejala')
        diagnosa = diagnosa_penyakit(gejala, model_cabai, gejala_columns_cabai)
        return render_template('diagnosis.html', penyakit=diagnosa, jenis_tanaman="Cabai")
    return render_template('gejala.html', jenis_tanaman="Cabai", gejala_list=gejala_columns_cabai)

@app.route('/gejala_tomat', methods=['GET', 'POST'])
def gejala_tomat():
    if request.method == 'POST':
        gejala = request.form.getlist('gejala')
        diagnosa = diagnosa_penyakit(gejala, model_tomat, gejala_columns_tomat)
        return render_template('diagnosis.html', penyakit=diagnosa, jenis_tanaman="Tomat")
    return render_template('gejala.html', jenis_tanaman="Tomat", gejala_list=gejala_columns_tomat)

@app.route('/daftarpenyakit_cabai')
def daftarpenyakit_cabai():
    penyakit_cabai = {
    "Layu Fusarium",
    "Layu Bakteri Ralstonia",
    "Buah Busuk Antraknosa",
    "Virus Kuning",
    "Bercak Daun (Cercospora sp.)"
}

    return render_template('daftarpenyakit.html', penyakit=penyakit_cabai, jenis_tanaman="Cabai")

@app.route('/daftarpenyakit_tomat')
def daftarpenyakit_tomat():
    penyakit_tomat = {
    "Busuk Daun",
    "Layu Bakteri",
    "Layu Fusarium",
    "Bercak Coklat",
    "Hawar Daun"
}
    return render_template('daftarpenyakit.html', penyakit=penyakit_tomat, jenis_tanaman="Tomat")

@app.route('/penyakit_cabai/<penyakit>')
def detail_penyakit_cabai(penyakit):
    detail = {
    "Layu Fusarium": {
        "gejala": ["Daun layu", "Daun menguning", "Batang rusak", "Batang menguning"],
        "solusi": [
            "Gunakan benih yang tahan terhadap fusarium",
            "Sterilisasi media tanam dan alat pertanian sebelum digunakan",
            "Lakukan rotasi tanaman dengan tanaman yang tidak rentan terhadap fusarium"
        ],
        "rekomendasi_produk": ["Fungi-X", "Fusarium Blocker", "Bio-Fertilizer A"]
    },
    "Layu Bakteri Ralstonia": {
        "gejala": ["Daun layu", "Batang rusak", "Batang menguning"],
        "solusi": [
            "Sanitasi lahan dengan solarisasi atau disinfektan tanah",
            "Hindari penggunaan air irigasi yang terkontaminasi",
            "Rotasi tanaman dengan jenis non-inang seperti jagung atau padi"
        ],
        "rekomendasi_produk": ["Ralsto-Killer", "BactoClean", "BioSanitizer"]
    },
    "Buah Busuk Antraknosa": {
        "gejala": ["Buah mengering", "Buah bercak mengkilap", "Buah busuk"],
        "solusi": [
            "Hindari penyiraman berlebihan yang membuat tanah terlalu lembab",
            "Buang dan bakar buah yang terinfeksi untuk mencegah penyebaran",
            "Gunakan varietas yang tahan antraknosa"
        ],
        "rekomendasi_produk": ["Anthra-Stop", "FruitShield", "Fungicide-X"]
    },
    "Virus Kuning": {
        "gejala": ["Daun menguning"],
        "solusi": [
            "Gunakan perangkap kuning untuk mengontrol populasi kutu kebul",
            "Tutup tanaman dengan jaring pelindung (paranet)",
            "Gunakan benih tahan virus atau varietas yang lebih toleran"
        ],
        "rekomendasi_produk": ["YellowTrap", "Net-Protect", "VirusSafe"]
    },
    "Bercak Daun (Cercospora sp.)": {
        "gejala": ["Daun coklat", "Daun rontok", "Tidak berbuah"],
        "solusi": [
            "Lakukan sanitasi lahan secara teratur dengan menghilangkan sisa tanaman",
            "Gunakan fungisida berbasis tembaga atau sulfur",
            "Tingkatkan sirkulasi udara di sekitar tanaman"
        ],
        "rekomendasi_produk": ["CercaShield", "LeafProtect", "CopperFungicide"]
    }
}
    return render_template('detailpenyakit.html', penyakit=penyakit, detail=detail[penyakit], jenis_tanaman="Cabai")

@app.route('/penyakit_tomat/<penyakit>')
def detail_penyakit_tomat(penyakit):
    detail = {
    "Busuk Daun": {
        "gejala": ["Daun bercak coklat", "Bawah daun bercak putih", "Batang berwarna coklat"],
        "solusi": [
            "Hindari penyiraman berlebihan dan jaga kelembaban tanaman",
            "Gunakan fungisida berbahan aktif tembaga atau mankozeb",
            "Lakukan pemangkasan pada daun yang terinfeksi"
        ],
        "rekomendasi_produk": ["FungiTom-X", "CopperShield", "LeafGuard"]
    },
    "Layu Bakteri": {
        "gejala": ["Tanaman layu keseluruhan", "Daun muda layu", "Tangkai daun merunduk"],
        "solusi": [
            "Sanitasi lahan dengan solarisasi atau penggunaan disinfektan tanah",
            "Gunakan varietas tahan layu bakteri",
            "Rotasi tanaman dengan jenis non-inang"
        ],
        "rekomendasi_produk": ["BactoStop", "SoilSanitizer", "HealthyPlant"]
    },
    "Layu Fusarium": {
        "gejala": ["Daun tua menguning", "Batang berwarna coklat", "Batas atas mengering"],
        "solusi": [
            "Sterilisasi alat pertanian sebelum digunakan",
            "Rotasi tanaman dengan jenis yang tidak rentan terhadap fusarium",
            "Gunakan fungisida sistemik yang sesuai"
        ],
        "rekomendasi_produk": ["FusariumBlock", "RootProtector", "BioFungicide"]
    },
    "Bercak Coklat": {
        "gejala": ["Daun bercak coklat", "Bawah daun bercak putih", "Tangkai daun berwarna putih"],
        "solusi": [
            "Hindari penyiraman daun untuk mengurangi kelembaban berlebih",
            "Gunakan fungisida berbasis klorotalonil atau tembaga",
            "Buang daun yang terinfeksi untuk mencegah penyebaran"
        ],
        "rekomendasi_produk": ["SpotClean", "LeafClear", "CopperProtect"]
    },
    "Hawar Daun": {
        "gejala": ["Daun muda layu", "Tangkai daun merunduk", "Tanaman layu keseluruhan"],
        "solusi": [
            "Jaga kebersihan lahan dengan membuang sisa-sisa tanaman",
            "Gunakan varietas tahan penyakit",
            "Lakukan pengendalian serangga vektor penyakit"
        ],
        "rekomendasi_produk": ["HawarStop", "BioControl", "DiseaseShield"]
    }
}
    return render_template('detailpenyakit.html', penyakit=penyakit, detail=detail[penyakit], jenis_tanaman="Tomat")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
