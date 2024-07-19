# KONFIGURASI FLASK PYTHON
from flask import Flask, render_template, request, redirect, flash, session

# LIBRARY PYTHON
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


app = Flask(__name__)
app.secret_key = 'super secret'

# PATH AND READ DATA
file_path = 'Data latih (prediksi kualitas tidur) new.xlsx'
DataFrame = pd.read_excel(file_path)

# DROP USELESS COLUMN
DataFrame_clc = DataFrame.drop(columns=['No'])

# CHANGE DTYPE FROM OBJECT TO INT
DataFrame_clc['Kualitas tidur'] = DataFrame_clc['Kualitas tidur'].map({'Baik': 0, 'Buruk': 1})

# PREPROCESSING
x = DataFrame_clc.drop(['Kualitas tidur'], axis=1)
y = DataFrame_clc['Kualitas tidur']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# MODELLING
# Daftar nilai K yang akan diuji
k_values = list(range(2, 31)) # 2 - 31

# Melakukan validasi silang luar untuk setiap nilai K
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Lipatan luar
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)  # Lipatan dalam

param_grid = {'n_neighbors': [i for i in range(2, 31)]}
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=inner_cv,
    scoring='accuracy'
)
grid_search.fit(x_train_smote, y_train_smote)

best_model = grid_search.best_estimator_
scores = cross_val_score(
    best_model, X=x_train_smote,
    y=y_train_smote,
    cv=outer_cv,
    scoring='accuracy'
)
tmp_ = pd.DataFrame(grid_search.cv_results_)

# Mendapatkan nilai K terbaik dan akurasi rata-rata terbaik dari NCV
best_k = tmp_.sort_values(by='rank_test_score')['param_n_neighbors'].iloc[0]
best_score = tmp_.sort_values(by='rank_test_score')['mean_test_score'].iloc[0]

# SAVED MODEL
best_knn_smote = KNeighborsClassifier(n_neighbors=best_k)
best_knn_smote.fit(x_train_smote, y_train_smote)

@app.route('/')
def index():
    session.pop('jawaban', None)
    return render_template('landing.html')

@app.route('/hasil', methods=['GET'])
def hasil():
    if ('jawaban' not in session) or (len(session['jawaban']) < 17):
        flash('Anda belum menjawab pertanyaan!!!')
        return redirect('/')
    
    inputan = {
        'Waktu di atas kasur (rebahan sampai bangun pagi) (Jam)': [session['jawaban']['1']],
        'Waktu rebahan sebelum tertidur (Menit)': [session['jawaban']['2']],
        'Waktu tertidur (Jam)': [session['jawaban']['3']],
        'Tidak dapat tertidur dalam 30 menit': [session['jawaban']['4']],
        'bangun tengah malam atau dini hari': [session['jawaban']['5']],
        'bangun ke kamar mandi': [session['jawaban']['6']],
        'tidak dapat bernafas dengan baik saat tertidur': [session['jawaban']['7']],
        'batuk atau mendengkur': [session['jawaban']['8']],
        'kedinginan saat tertidur': [session['jawaban']['9']],
        'kepanasan saat tertidur': [session['jawaban']['10']],
        'terbangun karena mimpi buruk': [session['jawaban']['11']],
        'merasakan nyeri saat tertidur': [session['jawaban']['12']],
        'gangguan lain saat tertidur': [session['jawaban']['13']],
        'konsumsi obat tidur': [session['jawaban']['14']],
        'mengantuk pada saat beraktifitas': [session['jawaban']['15']],
        'memikirkan masalah saat beraktifitas': [session['jawaban']['16']],
        'kualitas tidur secara objektif': [session['jawaban']['17']],
    }

    inputan = pd.DataFrame(inputan)
    y_pred = best_knn_smote.predict(inputan)

    # Mendapatkan tetangga terdekat dari data latih setelah SMOTE
    distances_smote, indices_smote = best_knn_smote.kneighbors(inputan)

    # Menampilkan beberapa tetangga terdekat dalam format baris dan kolom
    neighbors_smote = []
    for i, idx in enumerate(indices_smote[0]):
        neighbor_data = x_train_smote.iloc[idx].copy()
        neighbor_quality = y_train_smote.iloc[idx]
        neighbors_smote.append({
            'index': i,
            'data': neighbor_data.to_dict(),
            'quality': 'Baik' if neighbor_quality == 0 else 'Buruk'
        })

    neighbors_smote_df = pd.DataFrame(neighbors_smote)
    
    label = {0: 'Baik', 1: 'Buruk'}

    inputan = inputan.rename(index={0: 'Inputan'})
    inputan = inputan.transpose()

    return render_template(
        'hasil.html',
        jawaban=session['jawaban'],
        pred=label[y_pred[0]],
        inputan=inputan.to_html(classes='table table-striped table-hover table-bordered table-sm table-responsive-sm'),
        neighbors=neighbors_smote_df.to_dict(orient='records'),
        best_k=best_k
    )

@app.route('/pertanyaan/<id>', methods=['GET', 'POST'])
def tanya(id):
    pertanyaan = {
        "1": 'Selama sebulan terakhir, berapa jam tiap malam yang anda habiskan di atas Kasur? (dari mulai rebahan di atas Kasur sampai bangun pagi)',
        "2": 'Selama sebulan terakhir, berapa lama (dalam menit) yang anda perlukan untuk dapat mulai tertidur setiap malam? (Waktu Yang Dibutuhkan Saat Mulai Berbaring Hingga Tertidur)',
        "3": 'Selama sebulan terakhir, berapa jam tiap malam yang anda habiskan saat tidur? (dari mulai tertidur hingga terbangun pada pagi hari)',
        "4": 'Selama sebulan terakhir, seberapa sering anda Tidak dapat tidur di malam hari dalam waktu 30 menit',
        "5": 'Selama sebulan terakhir, seberapa sering anda Bangun tengah malam atau dini hari',
        "6": 'Selama sebulan terakhir, seberapa sering anda Harus bangun untuk ke kamar mandi',
        "7": 'Selama sebulan terakhir, seberapa sering anda Tidak dapat bernafas dengan nyaman saat berbaring',
        "8": 'Selama sebulan terakhir, seberapa sering anda Batuk atau mendengkur keras',
        "9": 'Selama sebulan terakhir, seberapa sering anda terbangun karena merasa kedinginan saat sedang tidur',
        "10": 'Selama sebulan terakhir, seberapa sering anda terbangun karena merasa kepanasan saat sedang tidur',
        "11": 'Selama sebulan terakhir, seberapa sering anda anda terbangun karena mimpi buruk saat sedang tidur',
        "12": 'Selama sebulan terakhir, seberapa sering anda anda terbangun karena merasakan nyeri saat sedang tidur',
        "13": 'Selama sebulan terakhir, seberapa sering anda penyebab lain yang belum disebutkan di atas yang menyebabkan anda terganggu di malam hari dan seberapa sering anda mengalaminya? (Contoh: kram)',
        "14": 'Selama sebulan terakhir, seberapa sering anda minum obat tidur (diresepkan dokter atau dibeli sendiri)?',
        "15": 'Selama sebulan terakhir, seberapa sering anda mengalami rasa mengantuk saat beraktivitas pada siang hari',
        "16": 'Selama sebulan terakhir, Apakah terdapat masalah yang anda pikirkan saat sedang beraktifitas?',
        "17": 'Bagaimana kualitas tidur anda secara keseluruhan dalam waktu satu bulan terakhir?',
    }
    
    if int(id) == 1:
        session.pop('jawaban', None)

    if 'jawaban' not in session:
        session['jawaban'] = {}

    if request.method == 'POST':
        session['jawaban'][str(int(id)-1)] = request.form['jawaban']
        session.modified = True
        return redirect('/pertanyaan/' + str(id))
    
    elif request.method == 'GET':
        if int(id) > 17:
            return redirect('/hasil')
        
        pertanyaan = pertanyaan[id]
        id = int(id) + 1
        return render_template('base.html', pertanyaan=pertanyaan, id=id, jawaban=session['jawaban'], jumlah=len(session['jawaban']))

@app.route('/pertanyaan', methods=['GET', 'POST'])
def pertanyaan():
    if request.method == 'POST':
        data_1 = int(request.form['1'])
        data_2 = int(request.form['2'])
        data_3 = int(request.form['3'])
        data_4 = int(request.form['4'])
        data_5 = int(request.form['5'])
        data_6 = int(request.form['6'])
        data_7 = int(request.form['7'])
        data_8 = int(request.form['8'])
        data_9 = int(request.form['9'])
        data_10 = int(request.form['10'])
        data_11 = int(request.form['11'])
        data_12 = int(request.form['12'])
        data_13 = int(request.form['13'])
        data_14 = int(request.form['14'])
        data_15 = int(request.form['15'])
        data_16 = int(request.form['16'])
        data_17 = int(request.form['17'])

        return redirect('/')
    
    elif request.method == 'GET':
        return render_template('pertanyaan.html', pertanyaan="Selama sebulan terakhir, berapa jam tiap malam yang anda habiskan di atas Kasur? (dari mulai rebahan di atas Kasur sampai bangun pagi)")

@app.route('/dataset')
def dataset():
	# Ambil data dari dataset
	items = pd.read_excel(file_path)
	# items['Kualitas tidur'] = items['Kualitas tidur'].map({'Baik':0,'Buruk':1})
	items = items.drop(columns=['No'])

	# data
	items = items.to_html(classes='table table-striped table-hover table-bordered table-sm table-responsive text-xs')

	# tampilkan
	return render_template('dataset.html', items=items)

if __name__ == '__main__':
    app.run(debug=True)