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
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def plot_loss(history, elem):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title(elem)
    plt.axis([-5, 100, 0, 70])
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

train_label_Cu = train_features[['CuOut']].copy()
train_label_Cd = train_features[['CdOut']].copy()
train_label_Zn = train_features[['ZnOut']].copy()

test_label_Cu = test_features[['CuOut']].copy()
test_label_Cd = test_features[['CdOut']].copy()
test_label_Zn = test_features[['ZnOut']].copy()

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

cu_dnn_model = build_and_compile_model(normalizer)
cu_dnn_model.summary()
cd_dnn_model = build_and_compile_model(normalizer)
cd_dnn_model.summary()
zn_dnn_model = build_and_compile_model(normalizer)
zn_dnn_model.summary()

history_cu = cu_dnn_model.fit(
    train_features,
    train_label_Cu,
    validation_split=0.2,
    verbose=0, epochs=100)
cu_dnn_model.save('3cu_dnn_model_100')

history_cd = cd_dnn_model.fit(
    train_features,
    train_label_Cd,
    validation_split=0.2,
    verbose=0, epochs=100)
cd_dnn_model.save('3cd_dnn_model_100')

history_zn = zn_dnn_model.fit(
    train_features,
    train_label_Zn,
    validation_split=0.2,
    verbose=0, epochs=100)
zn_dnn_model.save('3zn_dnn_model_100')

plot_loss(history_cu, 'Cu')
plot_loss(history_cd, 'Cd')
plot_loss(history_zn, 'Zn')

#cu_dnn_model = tf.keras.models.load_model('cu_dnn_model_100')
#cd_dnn_model = tf.keras.models.load_model('cd_dnn_model_100')
#zn_dnn_model = tf.keras.models.load_model('zn_dnn_model_100')


test_results = {}
test_results['cu_dnn_model'] = cu_dnn_model.evaluate(test_features, test_label_Cu, verbose=0)
test_results['cd_dnn_model'] = cd_dnn_model.evaluate(test_features, test_label_Cd, verbose=0)
test_results['zn_dnn_model'] = zn_dnn_model.evaluate(test_features, test_label_Zn, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T