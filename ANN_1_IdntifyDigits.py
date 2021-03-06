%pylab inline
import os
import numpy as np
import pandas as pd
from scipy.misc import imread   ## Read an image from a file as an array.
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import tensorflow as tf
import keras

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)


seed =128
rng = np.random.RandomState(seed)

##  The first step is to set directory paths, for safekeeping!

root_dir = os.path.abspath("H:/DS/Python/ANN/DigitsIdentificationByImage/")
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))

sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

train.head()
test.head()

img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'Train', img_name)

img  = imread(filepath,flatten=True)

plt.imshow(img,cmap='gray')
plt.axis('off')
plt.show()

#The above image is represented as numpy array

img

#For easier data manipulation, let’s store all our images as numpy arrays


temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, 'Train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    print("Done With: "+(img_name))
    
train_x = np.stack(temp)

train_x /=255.0
train_x = train_x.reshape(-1, 784).astype('float32')

temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir,'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    print("Done With: "+(img_name))
    
test_x = np.stack(temp)

test_x /= 255.0
test_x = test_x.reshape(-1, 784).astype('float32')


train_y = keras.utils.np_utils.to_categorical(train.label.values)


split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]
train.label.ix[split_size:]



# define vars
input_num_units = 784
hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

# import keras modules

from keras.models import Sequential
from keras.layers import Dense


# Create Model

model = Sequential([
        Dense(output_dim=hidden_num_units,input_dim = input_num_units,activation='relu'),
        Dense(output_dim = output_num_units,input_dim = hidden_num_units,activation='softmax')
        ])

    
#compile the model with necessary attributes

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# run the Model

trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))


#STEP 3: Model Evaluation
#a) To test our model with our own eyes, let’s visualize its predictions

pred = model.predict_classes(test_x)

img_name = rng.choice(test.filename)
filepath = os.path.join(data_dir,'test', img_name)

img = imread(filepath, flatten=True)

test_index = int(img_name.split('.')[0]) - train.shape[0]

print ("Prediction is: ", pred[test_index])

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

sample_submission.filename = test.filename; sample_submission.label = pred
sample_submission.to_csv(os.path.join(sub_dir,'sub03.csv'),index=False)
