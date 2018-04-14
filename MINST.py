from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
features = train.drop(["label"], 1).values
features = StandardScaler().fit_transform(features)
target = train['label'].values
target1 = train['label'].values
target = to_categorical(target) # doubt 
model = Sequential()
model.add(Dense(392, activation='relu', input_dim=784))
model.add(Dense(196, activation='relu'))
model.add(Dense(98, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
             optimizer='adam',
             metrics=['accuracy'])
model.fit(features,target, nb_epoch=30, batch_size=10,  verbose=2)
features_test = test.values
features_test = StandardScaler().fit_transform(features_test)
prediction = model.predict_classes(features_test, verbose = 1)
my_solution = pd.DataFrame({'ImageId':list(range(1, len(prediction)+1)), 'Label': prediction })
print(my_solution)
my_solution.to_csv("submission.csv", index = False)