#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pickle


# In[38]:


model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(4,(3,3),input_shape=(50,50,3),activation='relu'),
            tf.keras.layers.Conv2D(8,(3,3),activation='relu'),
            tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
            tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64,activation='relu'),
            tf.keras.layers.Dense(32,activation='relu'),
            tf.keras.layers.Dense(16,activation='relu'),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.summary()
model.compile(optimizer='adam',metrics=['accuracy'],loss = 'binary_crossentropy')


# In[39]:


import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "dataset_final/train_data/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
)

VALIDATION_DIR = "dataset_final/test_data/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(50,50),
	class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(50,50),
	class_mode='binary'
)


# In[40]:

#test_labels = pickle.load(open( "testlabels", "rb" ))
#train_labels = pickle.load(open("trainlabels", "rb"))


print('#' * 10)


#print(train_generator)
#type(train_generator)

#print(validation_generator)
#print(type(validation_generator))
#print(test_labels)
#print(type(test_labels))
print('####################')
#print(train_labels)

history =  model.fit(train_generator, epochs=25, validation_data = validation_generator)

test_loss, test_acc = model.evaluate(validation_generator, verbose=2)

print('Metrics')
print('Test Accuracy', test_acc)

model.save('/media/sf_New_folder/extra_layer_2_cancer.h5')
#ff=open("/media/sf_New_folder/model_cancer.pkl","wb")
#pickle.dump(model,ff)
#ff.close()

# In[ ]:




