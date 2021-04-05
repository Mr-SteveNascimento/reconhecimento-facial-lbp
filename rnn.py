# importando os pacotes necessários
import itertools
import os
from sklearn.model_selection import train_test_split
from localbinarypatterns import LocalBinaryPatterns
from sklearn.neighbors import RadiusNeighborsClassifier
from imutils import paths
import cv2
import pandas as pd
import numpy as np
import time
from PIL import Image

inicio = time.time()

def tulis_hasil(hasil, fname):
    hasil = np.array(hasil)
    pd.DataFrame({"r": hasil[:, 0], "points": hasil[:, 1], "radius": hasil[:, 2],
                  "akurasi": hasil[:, 3]}).to_csv(fname, index=False, header=True)

db = 'gt_db'
data_training = 'db/%s' % db

points = range(8, 70, 4)
radius = range(4, 21)
par = list(itertools.product(points, radius))
hasil = []

dataset = [file for file in paths.list_images(data_training)]

# Essa é nossa contribuição
count = 0
for imagePath in paths.list_images(data_training):
    count += 1
    size = 150, 113
    im = Image.open(imagePath)
    width, height = im.size
    if(width!=150 and height!=113):
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save(str(imagePath)[:13]+str(count)+".jpg", "JPEG")
        os.remove(imagePath, dir_fd=None)

for p in par:
    print("##### points: %d, radius:%d #####" % p)
    # inicializar o descritor de padrões binários locais
    desc = LocalBinaryPatterns(p[0], p[1])
    data = []
    labels = []

    # loop nas imagens de treinamento
    print(data_training, len(dataset))
    for imagePath in paths.list_images(data_training):
        # carregue a imagem, converta-a em tons de cinza e descreva-a
        # print("test:", imagePath)
        image = cv2.imread(imagePath)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            print('ERROR di:', imagePath)
        hist = desc.describe(gray)
        # extraia o rótulo do caminho da imagem e atualiza o
        # rótulo e listas de dados
        # print(imagePath.split("/"))  # use "\\" no Windows
        labels.append(imagePath.split("/")[-2])  # use "\\" no Windows
        data.append(hist)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=0)

    # treinar um RNN Linear nos dados
    r_nn = [0.005, 0.01, 0.015, 0.02]
    for r in r_nn:
        # neigh = RadiusNeighborsClassifier(radius=k, outlier_label=0.1)
        # outlier_label=0.1
        '''
        outlier_label{manual label, ‘most_frequent’}, default=None
            label for outlier samples (samples with no neighbors in given radius).
            * manual label: str or int label (should be the same type as y) or list of manual labels if multi-output is used.
            * ‘most_frequent’ : assign the most frequent label of y to outliers.
            * None : when any outlier is detected, ValueError will be raised.
        '''
        neigh = RadiusNeighborsClassifier(radius=r, outlier_label='most_frequent')
        neigh.fit(X_train, y_train)

        benar = 0
        jml = 0
        for i in range(len(y_test)):
            jml += 1
            hist = X_test[i]
            prediction = neigh.predict([hist])[0]
            if prediction == y_test[i]:
                benar += 1
        akurasi = float(benar*100/jml)
        print(benar, jml, r, ": Akurasi", akurasi, "%")
        hasil.append([r, p[0], p[1], akurasi])
tulis_hasil(hasil, "results/{0}_rnn.csv".format(db))


fim = time.time()
print(fim-inicio)
