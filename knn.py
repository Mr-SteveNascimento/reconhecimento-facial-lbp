# importando os pacotes necessários
import itertools
import os
from sklearn.model_selection import train_test_split
from localbinarypatterns import LocalBinaryPatterns
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import cv2
import pandas as pd
import numpy as np
import time
from PIL import Image

inicio = time.time()

def tulis_hasil(hasil, fname):
    hasil = np.array(hasil)
    pd.DataFrame({"k": hasil[:, 0], "points": hasil[:, 1], "radius": hasil[:, 2],
                  "akurasi": hasil[:, 3]}).to_csv(fname, index=False, header=True)


db = 'gt_db'
data_training = 'db/%s' % db

points = range(8, 70, 4)
radius = range(4, 21)
par = list(itertools.product(points, radius))
hasil = []

dataset = [file for file in paths.list_images(data_training)]

# count = 0
# for imagePath in paths.list_images(data_training):
#     count += 1
#     size = 150, 113
#     im = Image.open(imagePath)
#     width, height = im.size
#     if(width!=150 and height!=113):
#         im_resized = im.resize(size, Image.ANTIALIAS)
#         im_resized.save("db/gt_db/s01/"+str(count)+".jpg", "JPEG")
#         os.remove(imagePath, dir_fd=None)

for p in par:
    print("##### points: %d, radius:%d #####" % p)
    # inicializar o descritor de padrões binários locais
    desc = LocalBinaryPatterns(p[0], p[1])
    data = []
    labels = []

    # loop nas imagens de treinamento
    print(data_training, len(dataset))
    for imagePath in paths.list_images(data_training):
        # carregue a imagem, convertendo em tons de cinza
        # print("test:", imagePath)
        image = cv2.imread(imagePath)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print('ERROR di:', imagePath)
        hist = desc.describe(gray)
        # extraia o rótulo do caminho da imagem e atualiza o
        # listas de dados
        # print(imagePath.split("/"))  # use "\\" no Windows
        labels.append(imagePath.split("/")[-2])  # use "\\" no Windows
        data.append(hist)
    # print(labels)
    # print('hist:', len(hist))
    # print(hist)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=0)

    # treinar um KNN Linear nos dados
    k_nn = range(1, 10, 2)
    for k in k_nn:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)

        benar = 0
        jml = 0
        for i in range(len(y_test)):
            jml += 1
            hist = X_test[i]
            prediction = neigh.predict([hist])[0]
            if prediction == y_test[i]:
                benar += 1
        akurasi = float(benar * 100 / jml)
        print(benar, jml, k, ": Akurasi", akurasi, "%")
        hasil.append([k, p[0], p[1], akurasi])
tulis_hasil(hasil, "results/{0}_knn.csv".format(db))

fim = time.time()
print(fim-inicio)
