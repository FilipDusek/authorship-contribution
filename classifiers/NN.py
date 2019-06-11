from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import os

# Disable tf debug information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NNClassifier:
    def __init__(self):
        self.clf = keras.models.Sequential()

    def fit(self, X, y):
        self.encoder = LabelEncoder().fit(y)
        y_encoded = self.encoder.transform(y)
        self.clf.add(keras.layers.Dense(1000, input_dim=X.shape[1], activation="sigmoid"))
        self.clf.add(keras.layers.Dropout(0.3, noise_shape=None, seed=1))
        self.clf.add(keras.layers.Dense(500, activation='sigmoid'))
        self.clf.add(keras.layers.Dropout(0.3, noise_shape=None, seed=2))
        self.clf.add(keras.layers.Dense(20, activation='softmax'))
        self.clf.compile(loss='sparse_categorical_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
        self.clf.fit(X, y_encoded, epochs=80, batch_size=80, verbose=0)

        return self

    def predict(self, X):

        return self.encoder.inverse_transform(self.clf.predict_classes(X))
