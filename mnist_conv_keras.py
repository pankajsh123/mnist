from keras.models import Sequential
from keras.layers import Dense, Activation , core , Convolution2D, MaxPooling2D , Dropout, Flatten , normalization
from keras.utils import np_utils
from keras.models import model_from_json

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
y_train = np_utils.to_categorical(y_train,10)
x_test = x_test.reshape(10000,28,28,1)
y_test = np_utils.to_categorical(y_test,10)

model = Sequential()

#first conv layer with 32 filters output (32,14,14)
model.add(Convolution2D(32,(2,2),padding = 'same',data_format = 'channels_last',input_shape = (28,28,1)))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2),padding = 'same',data_format = 'channels_last'))
model.add(core.Dropout(0.25))

#second conv layer with 64 filter output (64,7,7)
model.add(Convolution2D(64,(2,2),padding = 'same',data_format = 'channels_last'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2),padding = 'same',data_format = 'channels_last'))
model.add(core.Dropout(0.25))

#fully connected layer with output 128
model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.25))

#output layer with output 10 classes
model.add(Dense(10,activation = 'softmax'))

#compile model on train dataset and check on test dataset
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 1,epochs = 1,validation_data = (x_test,y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(model.layers)

#save model for future use in json format
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")




