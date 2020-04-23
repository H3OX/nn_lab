import joblib
import datetime
import numpy as np
import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

r = joblib.load('../data/r.pkl')
s = joblib.load('../data/s.pkl')
t = joblib.load('../data/t.pkl')
d = r.append(s, ignore_index=True)
q = d.append(t, ignore_index=True)

X = np.expand_dims(q.iloc[:, 1:].values, axis=2)
y = k.utils.to_categorical(q.iloc[:, 0].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)

inputs = k.Input(shape=(30, 1))
x = k.layers.BatchNormalization()(inputs)
x = k.layers.LSTM(30, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
x = k.layers.LSTM(30, activation='sigmoid', dropout=0.3, recurrent_dropout=0.2, return_sequences=True)(x)
x = k.layers.Flatten()(x)
outputs = k.layers.Dense(7, activation='softmax')(x)


model = Model(inputs, outputs, name='LSTM_model')
model.compile(optimizer=Adam(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = k.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])
