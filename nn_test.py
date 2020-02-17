import joblib
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


dataset = joblib.load('data/tess_rvds.pkl')
X = np.expand_dims(dataset.iloc[:, 1:].values, axis=2)
y = to_categorical(dataset.iloc[:, 0].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
# Model definition
adam = Adam(learning_rate=0.0005)
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=3, input_shape=(30, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(rate=0.2))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))

model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(rate=0.2))
model.add(Flatten())

model.add(Dropout(rate=0.2))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

training = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
val_loss = training.history['val_loss']
loss = training.history['loss']
acc = training.history['accuracy']
val_acc = training.history['val_accuracy']

plt.figure()
plt.plot(val_acc, c='g')
plt.plot(acc, c='r')
plt.title('Val-acc vs acc')
plt.show()

plt.figure()
plt.plot(val_loss, c='g')
plt.plot(loss, c='r')
plt.title('Val-loss vs loss')
plt.show()