from keras.models import Sequential 
from keras.models import Convolution2D
from keras.models import MaxPooling2D
from keras.models import Flatten
from keras.models import Dense

classifier = Sequential()

classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
