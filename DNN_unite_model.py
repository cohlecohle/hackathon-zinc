import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import keras
from keras.models import Sequential
import tensorflow as tf

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3)
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.axis([-5, 30, 0, 50])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.savefig('testfigure.png', dpi=100)
    plt.show()

np.set_printoptions(precision=3, suppress=True)

raw_dataset = pd.read_csv('data.csv')
dataset = raw_dataset.copy()
dataset = dataset.drop(columns=['Unnamed: 0', 'DateTime'])
dataset.head()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[["Sol", "CuIn", "CdIn", "ZnIn", "Temperature", "pH", "Dust"]], diag_kind='kde')

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features[['CuOut', 'CdOut', 'ZnOut']].copy()
test_labels = test_features[['CuOut', 'CdOut', 'ZnOut']].copy()

train_features = train_features.drop(['CuOut', 'CdOut', 'ZnOut'], axis=1)
test_features = test_features.drop(['CuOut', 'CdOut', 'ZnOut'], axis=1)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=25)

plot_loss(history)

print(123)