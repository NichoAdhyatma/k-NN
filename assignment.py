from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Classification using k-NN

# gunakan dataset sesuai soal

# dataset = pd.read_csv('milk.csv')
# dataset = pd.read_csv('milk_training.csv')
# dataset = pd.read_csv('milk_testing.csv')

# train_data = np.array(dataset)[:, 1:-1]
# train_label = np.array(dataset)[:, -1]

# test_data = np.array(dataset)[:, 1:-1]
# test_label = np.array(dataset)[:, -1]

# Menampilkan train_data & train_label
# print("Train data : \n", train_data)
# print("\nTrain label : \n", train_label)

# Menampilkan test_data & test_label
# print("Test data : \n", test_data)
# print("\nTest label : \n", test_label)

# Normalisasi train_data
# sc = MinMaxScaler(feature_range=(0, 1))
# data = sc.fit_transform(train_data)
# print("\nNormalisasi min_max (0, 1) :  \n", data)

# Normalisasi test_data
# sc = MinMaxScaler(feature_range=(0, 1))
# data = sc.fit_transform(test_data)
# print("\nNormalisasi min_max (0, 1) :  \n", data)

# Klasifikasi menggunakan k-NN untuk 1 input data untuk train_data & train_label

# kNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
# kNN.fit(train_data, train_label)

# temperature = input('Masukkan Temperature: ')
# taste = input('Masukkan Taste: ')
# odor = input('Masukkan Odor: ')
# fat = input('Masukkan Fat: ')
# turbidity = input('Masukkan Turbidity: ')
# color = input('Masukkan Colour: ')
# test_data = np.array([int(temperature), int(taste), int(odor), int(fat), int(turbidity), int(color)])
# test_data = np.reshape(test_data, (1, -1))
# print("Test Data \n", test_data)
# hasil = kNN.predict(test_data)
# print("Hasil dari k-NN : ", hasil)


# Membandingkan klasifikasi k-NN test_data & test_label

# kNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
# kNN.fit(test_data, test_label)
#
# temperature = input('Masukkan Temperature: ')
# taste = input('Masukkan Taste: ')
# odor = input('Masukkan Odor: ')
# fat = input('Masukkan Fat: ')
# turbidity = input('Masukkan Turbidity: ')
# color = input('Masukkan Colour: ')
# test_data = np.array([int(temperature), int(taste), int(odor), int(fat), int(turbidity), int(color)])
# test_data = np.reshape(test_data,(1,-1))
# print(test_data)
# hasil = kNN.predict(test_data)
# print("Hasil dari k-NN : ", hasil)
