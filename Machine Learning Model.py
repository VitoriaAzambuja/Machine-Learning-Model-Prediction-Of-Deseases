import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.datasets import load_iris

#reading csv file
data = pd.read_csv('heart_failure.csv')

#Print columns and df Types
print(data.info())

#Print distribution of death events
from collections import Counter
print('Classes and number of values in the dataset',Counter(data['death_event']))

#Reassing results from death events to variables
y = data["death_event"]
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

#Converting categorical features to one-hot enconfind vector
x  = pd.get_dummies(x)

#Splitting data into training and test features and labels
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#Scaling numeric features into dataset
from sklearn.preprocessing import Normalizer
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

#Training scaler instance
X_train = ct.fit_transform(X_train)

#Scalling test data usign the trained scaler
X_test = ct.transform(X_test)

#Preparing labels for classification
le = LabelEncoder()

#Fitting the encoder to training label and converting training label to trained encoder
Y_train = le.fit_transform(Y_train.astype(str))

#Enconding test labels using trained enconder
Y_test = le.transform(Y_test.astype(str))

#Transforming encoded training label into binary vector
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train)

#Enconding test label into binary vector
from tensorflow.keras.utils import to_categorical
Y_test = to_categorical(Y_test)

#Designing model
from tensorflow.keras.models import Sequential
model = Sequential()

model.add(InputLayer(input_shape=(X_train.shape[1],)))

from tensorflow.keras.layers import Dense
model.add(Dense(12, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training and evaluating the model
model.fit(X_train, Y_train, epochs = 50, batch_size = 2, verbose=1)

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Loss", loss, "Accuracy:", acc)

#Generating Classification Report
y_estimate = model.predict(X_test, verbose=0)

import numpy as np
y_estimate = np.argmax(y_estimate, axis=1)

import numpy as np
y_true = np.argmax(Y_test, axis=1)

print(classification_report(y_true, y_estimate))

print(data)