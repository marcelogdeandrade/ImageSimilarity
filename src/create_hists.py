import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pickle
import matplotlib.pyplot as plt

NUMBER_IMAGES = 50

DIR = 'images/'
files = []
for root, directories, filenames in os.walk(DIR):
    for filename in filenames: 
        files.append(os.path.join(root,filename))

def computa_descritores(img):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    return des

def cria_vocabulario(descritores, sz):
    scaler = StandardScaler()
    descs = []
    for line in descritores:
        for d in line[1]:
            descs.append(d)
    norm_data = scaler.fit_transform(descs)
    kmeans = KMeans(n_clusters=sz)
    kmeans.fit(norm_data)
    return (kmeans.cluster_centers_, kmeans, scaler)

def representa_histograma(img, vocab):
    scaler = vocab[2]
    descs = computa_descritores(img)
    norm_descs = scaler.transform(descs)
    kmeans = vocab[1]
    cluster = vocab[0]
    pred = kmeans.predict(norm_descs)
    # Create hist
    hist = [0] * len(cluster)
    for i in pred:
        hist[i] += 1 / len(pred)
    return hist

def cria_histogramas(files, size_vocab = 300):
    result_descs = []
    for i in files:
        img = cv2.imread(i,0)
        descs = computa_descritores(img)
        result_descs.append((i, descs))
    vocab = cria_vocabulario(result_descs, size_vocab)
    hists = []
    for i in files:
        img = cv2.imread(i,0)
        hist = representa_histograma(img, vocab)
        hists.append((i, hist))
    return (hists, vocab)


hists, vocab = cria_histogramas(files[:NUMBER_IMAGES])

with open('hists.pkl', 'wb') as f:
    pickle.dump(hists, f)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
