from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# Classification using k-NN

dataset = pd.read_csv('ruspini.csv')
train_data = np.array(dataset)[:, 1:-1]
train_label = np.array(dataset)[:, -1]

kNN = KNeighborsClassifier(n_neighbors=3, weights='distance')
kNN.fit(train_data, train_label)

# Input 1 Data & Testing
# angkaX = input('Masukkan X: ')
# angkaY = input('Masukkan Y: ')
# test_data = np.array([int(angkaX), int(angkaY)])

# test_data=np.reshape(test_data,(1,-1))
# print(test_data)
# hasil=kNN.predict(test_data)
# print("Hasil dari k-NN : ",hasil)

# Testing lebih dari 1 Data
hasil = kNN.predict(train_data)
print("Test Label : ", train_label)
print("Hasil k-NN : ", hasil)
