# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:22:02 2024

@author: Furkan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import os
import gc
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

# Resim dosyalarının bulunduğu dizin
path = "images"

# Eğitim setini yükle
df = pd.read_csv('train.csv', dtype={'image': 'object', 'category': 'int8'})

# Gemi türleri sözlüğü
ship = {'Cargo': 1, 'Military': 2, 'Carrier': 3, 'Cruise': 4, 'Tankers': 5}

# Gemi türleri sözlüğünü ters çevir
ship = dict([[v, k] for k, v in ship.items()])

# Test etiketlerini yorumlanabilir hale getir
df['ship'] = df['category'].map(ship).astype('category')
labels = list(df['ship'].unique())

# Gemi türlerinin dağılımını göster
print(df['ship'].value_counts(normalize=True))

# OneHotEncoder kullanarak kategorileri dönüştür
ohe = OneHotEncoder(dtype='int8', sparse=False)
y_train = ohe.fit_transform(df['category'].values.reshape(-1, 1))




def load_images(target_size=(224, 224)):
    array = []
    for file in tqdm(df['image'].values):
        img_path = os.path.join(path, file)
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=target_size)
            img = img_to_array(img) / 255.0  # normalize image tensor
            array.append(img)
        else:
            print(f"Dosya bulunamadı: {img_path}")
    gc.collect()
    return np.asarray(array)




# Görüntüleri yükle
X_train = load_images(target_size=(224, 224))

print(X_train.shape)
print(y_train.shape)


    
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting normalize=True.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    gc.collect()
    
    
   

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                    stratify=y_train,
                                                    random_state=42,
                                                    test_size=0.2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  stratify=y_train,
                                                  random_state=42,
                                                  test_size=0.25)

from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
best = load_model('best_xception.hdf5')

# Test setinde tahmin yap
y_test_pred = np.argmax(best.predict(X_test), axis=1)
true_test = np.argmax(y_test, axis=1)

# Karışıklık matrisi oluşturma
cm = confusion_matrix(true_test, y_test_pred)



# Karışıklık matrisini görselleştirme
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Sınıf isimlerini liste olarak al
labels = list(df['ship'].unique())

# Karışıklık matrisini çiz ve kaydet
plot_confusion_matrix(cm, classes=labels, normalize=True, title='Confusion Matrix (Test)')

# Save confusion matrix plot for test set
plt.savefig('confusion_matrix_test.png')

# F1 Skorunu yazdır
test_f1_score = f1_score(true_test, y_test_pred, average="weighted")
print(f'Weighted F1 Score (Test): {test_f1_score:.4f}')