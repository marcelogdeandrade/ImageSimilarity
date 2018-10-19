import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import pickle
import matplotlib.pyplot as plt

IMG_PATH = "test/image1.png"

with open('hists.pkl', 'rb') as f:
    hists = pickle.load(f)
    
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

def computa_descritores(img):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    return des

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

img1 = cv2.imread(IMG_PATH)

hist1 = representa_histograma(img1, vocab)

result = []
for hist2 in hists:
    compare = cv2.compareHist(np.array(hist1).astype(np.float32),np.array(hist2[1]).astype(np.float32),0)
    result.append((compare, hist2[0]))
result.sort(key=lambda tup: tup[0], reverse=True)

def show_imgs(img, imgs, num_imgs = 3):
    fig = plt.figure('original')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    for i in range(num_imgs):
        fig = plt.figure(f'top #{i+1}')
        ax = fig.add_subplot(1, 1, 1)
        img = cv2.imread(imgs[i][1])
        ax.imshow(img)
    plt.show()

show_imgs(img1, result)
