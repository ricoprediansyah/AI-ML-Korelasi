# ML-1 Korelasi

# Student yang pakai Colab, silahkan "Save a Copy in Drive" dahulu
# Student yang pakai Jupyter Notebook, silahkan download code ini dahulu

# Import library yang di buatuhkan

# import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import warnings
# warnings.filterwarnings("ignore")


# Uji Asumsi Klasik pada Regresi Linier
# Regresi Linier memiliki 5 asumsi yang perlu dipenuhi. Asumsi ini sebenarnya wajib dalam teori ilmu statistika, namun dalam praktik machine learning proses ini terkadang diabaikan karena ketidaktahuan. Uji ini berfungsi untuk ketepatan estimasi dan menjauhkan dari bias

# Hubungan (korelasi) linier antara feature dan label
# Uji Normalitas
# Tidak terjadi Heteroskedastisitas
# Tidak terdapat autokorelasi
# Tidak terdapat multikolinearitas (untuk regresi berganda)
# Sumber: buku Multivariate data analysis oleh Hair

# Download dataset
# ! wget -O 50_Startups.csv https://www.dropbox.com/s/z2ue4a1ogefcuo3/50_Startups.csv?dl=0

# Load Data
data = pd.read_csv('50_Startups.csv')

# Periksa info terkait dataset (metadata)
data.info()

# 1. Hubungan Linier antara tiap feature dengan label (X dan Y)
# Pertama kita akan menentukan mana variable independen (X) dan dependen (Y).
# Pada aktivitas ini, kita akan menggunakan kolom R&D Spend, Administration, dan Marketing Spend sebagai variable independen/feature.
# Sisanya, kolom Profit, akan kita gunakan sebagai variable dependen (Y).

# Selanjutnya, kita perlu melihat hubungan linear tiap X dengan Y.
# Hubungan linier bisa dilihat menggunakan analisis korelasi atau plot.

# Analisis korelasi
data.corr()

# visualisasi menggunakan heatmap
sns.heatmap(data.corr(), annot=True)
plt.show()

# Dari analisis korelasi terlihat variable yang punya hubungan linier cukup kuat dengan profit adalah

# Marketing Spend dan
# R&D Spend
# Mari lihat hasil scatter plot tiap kolom untuk lebih meyakinkan!

# Visualisasi hubungan antara marketing spend dan profit
plt.scatter(data['Marketing Spend'], data['Profit'], color='blue')
plt.xlabel('Marketting Spend')
plt.ylabel('Profit')
plt.show()

# Visualisasi hubungan antara R&D Spend dan profit
plt.scatter(data['R&D Spend'], data['Profit'], color='blue')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()

# Visualisasi hubungan antara Administration dan profit
plt.scatter(data['Administration'], data['Profit'], color='blue')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.show()

# Terlihat bahwa R&D dan marketing spend plotnya masih cenderung membentuk garis yang menunjukkan masing-masing memiliki hubungan yang kuat dengan profit.
# Sementara, administration plotnya terlihat berpencar sehingga tidak ada hubungan linier dengan profit.
# Jadi kedepannya variabel independen yang kita gunakan sebagai feature adalah Marketing Spend dan R&D Spend.

# Splitting Data Menjadi Data Train dan Test
# Berikut adalah programnya:

features = ['R&D Spend','Marketing Spend']
X = data[features].values
Y = data.Profit
X_train, X_test, Y_train, Y_test = train_test_split (X,Y, test_size=0.2, random_state=23)

# Periksa bentuk data train dan test
print('Data Train : ', end='')
print(X_train.shape, Y_train.shape)
print ('\nData Test : ', end='')
print(X_test.shape, Y_test.shape)

#Modeling
# Membuat model Regresi Linier (Linier Regression) kemudian melatihnya menggunakan data train.
lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

# 2. Uji normalitas
# Disini yang kita uji adalah normalitas residual (errornya).
# Mengapa? karena menurut sifat distribusi normal, kalau errornya normal dan modelnya linier maka koefisien regresinya juga berdistribusi normal (terhindar dari bias outlier).
# Berikut adalah programnya:

y_predtrain = lin_reg.predict(X_train)  #prediksi model data train
err = y_predtrain - Y_train             #residual data train
sns.distplot(err)                       #membuat plot histogram

z_er = stats.zscore(err)
norm_er = stats.kstest(z_er, 'norm', )  #uji kolmogrov-smirnov
print('hasil uji kolmogrov Smirnoc \n', norm_er)

# Ho : data = berdistribusi normal
# Ha : data tidak berdistribusi normal
# Karena pvalue > 0.05, maka Ho diterima.
# Plot histogram yang membentuk lonceng dan puncaknya cenderung ke tengah juga mendukung bahwa errornya sudah berdistribusi normal.


# 3. Tidak terjadi Heteroskedastisitas
# Homoskedastisitas adalah kondisi ketika nilai residu/error pada tiap nilai prediksi bervariasi dan variasinya cenderung konstan.
# Lawan dari homoskedastisitas adalah heteroskedastisitas, dimana error cenderung berubah dan malah berkorelasi dengan prediksinya. Jika ini terjadi tentu menunjukkan bahwa prediksi ini memiliki suatu bias tertentu.
# Untuk mengecek hal ini kita bisa lihat dari scatter plot error-nya.

plt.figure(figsize=(8, 4))
plt.scatter(y_predtrain, err)
plt.show()

# Terlihat plot error-nya berada di sekitar angka yang sama meski nilai prediksinya bertambah.
# Artinya nilai prediksi kita tidak terganggu oleh errornya atau tidak terjadi heteroskedastisitas.
# Berikut adalah contoh plot jika terjadi heteroskedastisitas, plotnya membentuk pola tertentu yang menunjukkan bahwa nilai prediksinya masih berkorelasi dengan error sehingga nilai prediksi kita masih terganggu oleh nilai errornya:
# E:\Aplikasi\vscode\code\download1.jpeg

# 4. Tidak terjadi Multikolinearitas (khusus regresi berganda)
# Multokolinearitas maksudnya adalah hubungan yang kuat antar feature.
# Regresi linier mengasumsikan bahwa feature-featurenya tidak saling berhubungan. Tentu saja ini hanya berlaku untuk regresi linier berganda (yang featurenya lebih dari 1).
# Salah 1 cara mengujinya adalah dengan melihat nilai VIF.

vif = [variance_inflation_factor (X_train, i) for i in range(len(X_train.T))]
pd.DataFrame({'VIF': vif[0:]}, index=features).T

# Some papers argue that a VIF<10 is acceptable, but others says that the limit value is 5.
# "10" as the maximum level of VIF (Hair et al., 1995)
# "5" as the maximum level of VIF (Ringle et al., 2015)

# Berdasarkan kriteria Hair, tidak terjadi kolinearitas.
# Catatan: Namun, berdasarkan kriteria Ringle masih terjadi kolinearitas. Disini kita menggunakan teori Hair.

# 5. Tidak terjadi Autokorelasi
# Autokorelasi adalah hubungan yang erat antar entry, misalnya antara data ke 4 dengan ke 5, data ke-6 dengan ke-7, dll.
# Autokorelasi juga harus dihindari dalam regresi linier. Uji ini tersedia di library yang lain, sehingga kita perlu training model lagi menggunakan library itu.

X_constant = sm.add_constant(X_train)         #ingat lagi 1x = R&D Spend, x2 = marketing spend
linreg = sm.OLS(Y_train, X_constant).fit()
linreg.summary()

# Uji Autokorelasi bisa kita lakukan menggunakan nilai Durbin-Watson (dw).
# Bandingkan nilai dw di atas dengan gambar di bawah!
# E:\Aplikasi\vscode\code\download2.jpeg
# Sebelum membandingan nilai dw dengan gambar di atas, kita perlu mengetahui nilai dL dan du terlebih dahulu.

# Nilai dL (lower bound) dan du (upper bound) bisa diketahui menggunakan DW table. Download DW table di sini.

# Diketahui:
# n = 50 (jumlah data)
# k = 2 (jumlah independen variable)
# dw = 2.147
# Maka:
# du = 1.628
# dL = 1.462
# Terakhir, bandingkan nilai dw dengan gambar di atas.

# E:\Aplikasi\vscode\code\download3.png
# Karena nilai dw (2.147) berada diantara nilai du (1.628) dan 4-du (2.372), maka tidak ada masalah autokorelasi.

# Catatan: Cara mengetahui nilai dL dan du menggunakan Table DW
# E:\Aplikasi\vscode\code\download4.png

# Mengevaluasi hasil regresi linier
# Kita akan mengevaluasi model menggunakan metrics MSE, RMSE, dan MAE
# Berikut adalah programnya:
y_predtest = lin_reg.predict(X_test) #prediksi data testing

# MSE
MSE_train = mean_squared_error(Y_train, y_predtrain)
print('Nilai MSE data training = ', MSE_train)
MSE_test = mean_squared_error(Y_test, y_predtest)
print('Nilai MSE data testing = ', MSE_test)

# RMSE
RMSE_train = np.sqrt(MSE_train)
print('Nilai RMSE data training = ', MSE_train)
RMSE_test = np.sqrt(MSE_test)
print('Nilai RMSE data testing = ', MSE_test)

# MAE
MAE_train = mean_absolute_error(Y_train, y_predtrain)
print('Nilai MAE data training = ', MAE_train)
MAE_test = mean_absolute_error(Y_test, y_predtest)
print('Nilai MAE data testing = ', MAE_test)

# Catatan: Untuk mengetahui apakah nilai ini cukup bagus atau tidak, kita perlu membuat model regresi lain lalu membandingkan MSE, RMSE, dan MAE-nya.
# Model terbaik adalah yang nilai MSE, RMSE, dan MAE-nya paling kecil

# Visualisasi Hasil Prediksi
# Kita akan memvisualisasikan hasil prediksi dengan data sebenarnya (data testing).
# Berikut ini adalah programnya:

#plotting data prediksi dan testing untuk membandingkan
plt.plot(y_predtest)
plt.plot(Y_test.values)

#untuk memberikan judul di grafik
plt.title('Prediction vs Real')

#menambahkan label y
plt.ylabel('Profit')

# menambahkan legend ke grafik
plt.legend(labels=['Prediction','Real'], loc='lower right')
#Terlihat bahwa nilai prediksi dan data testing cukup dekat.

# Koefisien Determinasi (R2)
print (f'R^2 score: {lin_reg.score(X,Y)}')

# Terlihat nilai R2 = 0.9499, ini merupakan nilai yang sangat bagus.
# Nilai ini menunjukkan 94.99% dari profit dapat diprediksi oleh R&D spend dan Marketing Spend.
# Sisanya (5.01%) dipengaruhi faktor lain yang tidak ada di model ini.

# Uji Simultan, Parsial, dan Besar Pengaruh Feature
linreg.summary()

## Uji Simultan
# Terlihat nilai p-value uji-F (Prob (F-statistic)) adalah 5.97 x 10^-26 < 0.05, artinya secara bersama-sama R&D Spend dan Marketing Spend berpengaruh signifikan terhadap Profit.
## Uji Parsial
# Terlihat nilai p-value uji-T (P>|t|) untuk R&D adalah 0.000 < 0.05 dan untuk Marketing 0.176, artinya secara sendiri-sendiri R&D Spend memberi pengaruh yang signifikan terhadap Profit, sementara pengaruh dari Marketing Spend tidak signifikan.
## Besar pengaruh feature
# Perhatikan kolom "coef", pada x1 (R&D Spend) nilainya 0.8251, artinya setiap perusahaan menaikkan kinerja R&D Spend 1 level saja mampu meningkatkan Profit perusahaan sebesar 0.8251. Sementara koefisien x2 (marketing) sebesar 0.0236. Artinya, selama ini pengaruh Marketing Spend terhadap Profit hanya 0.0236.

# Kesimpulan
# Karena model yang kita buat telah memenuhi Uji Asumsi Klasik, maka model tersebut sudah bisa kita pakai untuk memprediksi data baru.
# Model ini juga sudah bisa kita deploy. Materi deployment akan disampaikan saat AI Domain.

# Input data baru
RnD_Spend = float(input('Input R&D Spend \t= '))
Marketing_Spend = float(input('Input Marketing Spend\t= '))
data_baru = [[RnD_Spend, Marketing_Spend]]

# Prediksi data baru menggunakan model Regresi Linear
hasil_prediksi = lin_reg.predict(data_baru)
hasil_prediksi = float(hasil_prediksi)

# Cetak hasil prediksi (Profit)
print('\nPrediksi Profit yang akan didapat adalah', hasil_prediksi)