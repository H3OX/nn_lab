import joblib
import numpy as np
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset = joblib.load('data.pkl')
X = np.expand_dims(dataset.iloc[:, 1:].values, axis=2)
y = to_categorical(dataset.iloc[:, 0].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
# Model definition
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, input_shape=(30, 1), activation='relu'))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(rate=0.2))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))

model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

training = model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test))
print(training.history)
