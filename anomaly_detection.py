# -*- coding: utf-8 -*-
"""
  Name: Anomaly Detection for Anonymous Dataset
  Author: Pablo Reynoso
  Date: 2022-03-22
  Version: 1.0
"""

"""## 0) Libraries/Frameworks"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model

"""## 1) Understanding the Dummy Dataset"""

ae_data = pd.read_csv('./dummy_data.csv', header=None)
ae_data = ae_data.add_prefix('var_')
print(ae_data.shape)

ae_data.describe()

"""## 2) Labeling Anomalies in Dummy Dataset"""

# Outliers Presence in Dummy Dataset
ax1 = sns.boxplot(data=ae_data, orient="h", palette="Set2")
plt.title('Dummy Dataset Outliers')
data_ = ae_data.copy(deep=True)

# Finding Threshold for Variables Outliers %
print('var_0 - outliers %: '+str(data_[(data_['var_0'] > 166)].shape[0])+'/10,000')
print('var_1 - outliers %: '+str(data_[(data_['var_1'] > 217)].shape[0])+'/10,000')
print('var_2 - outliers %: '+str(data_[(data_['var_2'] > 149)].shape[0])+'/10,000')
print('var_3 - outliers %: '+str(data_[(data_['var_3'] > 166)].shape[0])+'/10,000')
print('var_4 - outliers %: '+str(data_[(data_['var_4'] > 197)].shape[0])+'/10,000')
print('var_5 - outliers %: '+str(data_[(data_['var_5'] > 81)].shape[0])+'/10,000')
print('var_6 - outliers %: '+str(data_[(data_['var_6'] > 202)].shape[0])+'/10,000')
print('var_7 - outliers %: '+str(data_[(data_['var_7'] > 184)].shape[0])+'/10,000')

# Finding the Anomalous Samples in Dummy Dataset (*considering 37.5% of anomaly in every sample)
var0_a = data_[(data_['var_0'] > 166)].index.tolist()
var1_a = data_[(data_['var_1'] > 217)].index.tolist()
var2_a = data_[(data_['var_2'] > 149)].index.tolist()
var3_a = data_[(data_['var_3'] > 166)].index.tolist()
var4_a = data_[(data_['var_4'] > 197)].index.tolist()
var5_a = data_[(data_['var_5'] > 81)].index.tolist()
var6_a = data_[(data_['var_6'] > 202)].index.tolist()
var7_a = data_[(data_['var_7'] > 184)].index.tolist()

outliers = var0_a + var1_a + var2_a + var3_a + var4_a + var5_a + var6_a + var7_a
anomalies = {x:outliers.count(x) for x in outliers}
anomalies = {k: v for k, v in sorted(anomalies.items(), key=lambda item: item[1], reverse=True)}
anomalies = {k: v for k, v in anomalies.items() if v > 2}
print(anomalies)

# Labeling Anomalous Samples
ae_data['normal'] = 1
ae_data.loc[list(anomalies.keys()), 'normal'] = 0
ae_data['normal']

ae_data

"""## 3) Splitting, Normalizing, Subsetting by Label *Dummy Dataset* for Supervised Anomaly Detection"""

# Dataframe to Numpy
raw_data = ae_data.values

# The last element contains the anomaly-labels
labels = raw_data[:, -1]

# The other data points are the dummy data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.20, random_state=9
)

# Normalize Data
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

# Filtering Train/Test into Normal & Anomalous Subsets
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

plt.grid()
plt.plot(np.arange(8), normal_train_data[0])
plt.title("A Normal Sample")
plt.show()

plt.grid()
plt.plot(np.arange(8), anomalous_train_data[0])
plt.title("An Anomalous Sample")
plt.show()

"""## 4) Autoencoder Architechture"""

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(8, activation="relu"),
      layers.Dense(4, activation="relu"),
      layers.Dense(2, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(2, activation="relu"),
      layers.Dense(4, activation="relu"),
      layers.Dense(8, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()

autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(normal_train_data, normal_train_data, 
          epochs=35, 
          batch_size=64,
          validation_data=(test_data, test_data),
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

"""## 5.1) Reconstruction Error - Normal Samples"""

encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(8), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

"""## 5.2) Reconstruction Error - Anomalous Samples"""

encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(8), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

"""## 6) Finding Threshold for Anomaly Detection"""

reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plt.hist(train_loss[None,:], bins=20)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plt.hist(test_loss[None, :], bins=20)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

"""## 7) Autoencoder Anomaly Classifier """

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)