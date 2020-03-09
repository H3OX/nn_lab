import joblib
import datetime
import numpy as np
import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

dataset = joblib.load('../data/tess_rvds_savee.pkl')
X = np.expand_dims(dataset.iloc[:, 1:].values, axis=2)
y = k.utils.to_categorical(dataset.iloc[:, 0].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

# Создание модели
inputs = k.Input(shape=(30, 1))
x = k.layers.Conv1D(filters=128, kernel_size=3)(inputs)
x = k.layers.BatchNormalization()(x)
x = k.layers.LeakyReLU()(x)
x = k.layers.Conv1D(filters=64, kernel_size=3)(x)
x = k.layers.LeakyReLU()(x)
x = k.layers.MaxPool1D(pool_size=2)(x)
x = k.layers.Dropout(rate=0.3)(x)
x = k.layers.BatchNormalization()(x)
x = k.layers.Conv1D(filters=32, kernel_size=3)(x)
x = k.layers.LeakyReLU()(x)
x = k.layers.Conv1D(filters=16, kernel_size=3)(x)
x = k.layers.MaxPool1D(pool_size=2)(x)
x = k.layers.Dropout(rate=0.3)(x)

x = k.layers.Flatten()(x)
x = k.layers.Dense(units=7)(x)
outputs = k.layers.Softmax()(x)


model = Model(inputs, outputs, name='cnn_model')
model.compile(optimizer=Adam(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

