import numpy as np
import tensorflow as tf
import tensorflow.keras
from KNN import KNN

(data_train, target_train), (data_test, target_test) = tensorflow.keras.datasets.mnist.load_data()

clf = KNN(7)
clf.fit(data_train, target_train)

for i in range(100):
    print("predict: ", clf._predict(data_test[i]))
    print("target: ", target_test[i])

