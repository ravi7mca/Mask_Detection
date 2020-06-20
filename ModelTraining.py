# Load data from directory

from keras.preprocessing.image import ImageDataGenerator

train_data_path = "train"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_path, shuffle=True, target_size=(200, 200), batch_size=20, class_mode='binary')

validation_dir = r"validation"
validation_generator = test_datagen.flow_from_directory(validation_dir, shuffle=True, target_size=(200, 200), batch_size=20, class_mode='binary')

# Model Training
from keras import layers
from keras import models
from keras import optimizers

#import datetime
model =  models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(200, 200, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss='binary_crossentropy', optimizer= optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#model.fit(X_train, y_train, epochs=10, batch_size=500)
#print(model.summary())

history = model.fit_generator(train_generator,steps_per_epoch=100, epochs=25, validation_data=validation_generator, validation_steps=40) # , callbacks=[tensorboard_callback])

# Displaying curves of loss and accuracy during training
import matplotlib.pyplot as plt
# %matplotlib inline
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#results = model.evaluate(X_test, y_test)
#results
#history.history
#print(model.summary())

model.save('Mask_on_Face_small_12_1.h5')