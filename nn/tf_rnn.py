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

inputs = k.Input(shape=(30, 1))
x = k.layers.BatchNormalization()(inputs)
x = k.layers.LSTM(30, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
x = k.layers.LSTM(30, activation='sigmoid', dropout=0.3, recurrent_dropout=0.2, return_sequences=True)(x)
x = k.layers.Flatten()(x)
outputs = k.layers.Dense(7, activation='softmax')(x)


model = Model(inputs, outputs, name='cnn_model')
model.compile(optimizer=Adam(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))