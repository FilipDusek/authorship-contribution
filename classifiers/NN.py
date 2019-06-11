from tensorflow import keras

class NNClassifier:
    def __init__(self):
        self.clf = keras.models.Sequential()

    def fit(self, X_train, y_train):
        self.clf.add(keras.layers.Dense(1000, input_dim=X_train.shape[1], activation = "relu"))
        self.clf.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
        self.clf.add(keras.layers.Dense(500, activation='sigmoid'))
        self.clf.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
        self.clf.add(keras.layers.Dense(100, activation='sigmoid'))
        self.clf.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
        self.clf.add(keras.layers.Dense(20, activation='sigmoid'))
        self.clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.clf.fit(X_train, y_train, epochs = 200, batch_size = 80, verbose = 0)


        return self

    def predict(self, X):
        return self.clf.predict_classes(X)