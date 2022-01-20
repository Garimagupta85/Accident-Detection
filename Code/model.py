import glob
import numpy as np
import random
from sklearn.model_selection import train_test_split 
import os
import numpy
import glob
import cv2
import random
from skimage import transform
import skimage
import sklearn
from sklearn.model_selection import train_test_split 

import os
import numpy as np

from tensorflow.python import keras 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
batch_size = 15
num_classes = 2
epochs = 30

row_hidden = 128
col_hidden = 128


frame , row, col =(99,144,256)

'''
	Loading all the Positive and negative according to the desired 
	file path and the load_set preprocess the data in which each frame
	is extracted to a paricular size of (144,256)

'''

def load_set(img_path):
	img = load_img(img_path)
	tmp = skimage.color.rgb2gray(np.array(img))
	tmp = transform.resize(tmp, (144, 256))
	return tmp

'''
	Loading all the Positive and negative according to the desired 
	file path and the load_set preprocess the data in which each frame
	is extracted to a paricular size of (144,256) and is horizontally filpped

'''

def horizontal_flip(img_path):
	img = load_img(img_path)
	tmp = skimage.color.rgb2gray(np.array(img))
	tmp = skimage.transform.resize(tmp, (144, 256))
	tmp = np.array(tmp)
	tmp = np.flip(tmp, axis = 1)
	return tmp


'''
	Loading all the Positive and negative files assigned to varaiable
	neg and pos respectively
	All files contains both the files paths

'''
pos = glob.glob( '99frames/*.mp4')
neg = glob.glob( 'negative/*.mp4')
all_files =  np.concatenate((pos, neg[0:len(pos)]))

#print(len(neg),len(pos))
#print(all_files)       


'''
	label matrix is used to make one hot encoding ie [0 1] for
	positve data and [1 0] for negative data

'''


def label_matrix(values):
    
    n_values = np.max(values) + 1    
    return np.eye(n_values)[values] 

labels = np.concatenate(([1]*len(pos), [0]*len(neg[0:len(pos)])))  
labels = label_matrix(labels)    
#print(len(labels))      





def load_data1(path):
	
	x = []
	for files in os.listdir(path):

		frames = []
		img_path = path+"/"+files
		if files !=("frame99.jpg"):
			img = load_set(img_path)
			x.append(img)
	return x

def load_data3(path):
	count = 0
	x = []
	for files in os.listdir(path):

		frames = []
		img_path = path+"/"+files
		if count < 99:
			count  = count + 1
			img = load_set(img_path)
			x.append(img)
	return x


def load_data2(path):	
	x = []
	
	for files in os.listdir(path):
		
		frames = []
		img_path = path+"/"+files
		if  files !=("frame99.jpg") :
			
			img = horizontal_flip(img_path)
			x.append(img)
	return x


def load_data4(path):	
	x = []
	count =0
	for files in os.listdir(path):
		
		frames = []
		img_path = path+"/"+files
		if  count < 99 :
			count = count +1
			img = horizontal_flip(img_path)
			x.append(img)
	return x	

'''

'''
def make_dataset(rand):
    seq1 = np.zeros((len(rand), 99, 144, 256))   
    for i,fi in enumerate(rand):                    
        print (i, fi)
                                      
        if fi[9:11] == '00' :
            t = load_data1(fi)
        elif fi[9:13] == 'MVIH':
        	t = load_data3(fi)
        elif fi[9:13] == 'MVI_':	
            t = load_data4(fi)                        
        elif fi[9:11]=='11' :
            t = load_data2(fi)         
                
        seq1[i] = t                                                                

    return seq1



x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.40, random_state=0) 
x_train = np.array(x_train); y_train = np.array(y_train)
                          ### need to be np.arrays
for i in range(0,len(x_train)):
	print(x_train[i],y_train[i])

x_testA = np.array(x_t1[:int(len(x_t1)/3)]); y_testA = np.array(y_t1[:int(len(x_t1)/3)])    #### test set
x_testA = make_dataset(x_testA)


### valid set for model
x_testB = np.array(x_t1[int(2*len(x_t1)/3):]); y_testB = np.array(y_t1[int(2*len(x_t1)/3):])    ### need to be np.arrays
x_testB = make_dataset(x_testB)

	


import keras
from keras.models import Model 
from keras.layers import Input,Dense,TimeDistributed
from keras.layers import LSTM

frame, row, col = (99, 144, 256)
x =Input(shape=(frame, row, col))
encoded_rows = TimeDistributed(LSTM(row_hidden))(x) 
encoded_columns =LSTM(col_hidden)(encoded_rows)

prediction = Dense(num_classes, activation='softmax')(encoded_columns)

model = Model(x, prediction)

model.compile(loss='categorical_crossentropy', 
				optimizer='NAdam',               
				metrics=['accuracy']) 

np.random.seed(18247)

import matplotlib.pyplot as plt
import numpy

for i in range(0, 30):               
    c = list(zip(x_train, y_train))  
    random.shuffle(c)                
    x_shuff, y_shuff = zip(*c)       
    x_shuff = np.array(x_shuff); y_shuff=np.array(y_shuff) 
    
    x_batch = [x_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)] 
    y_batch = [y_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)] 

    for j,xb in enumerate(x_batch): 
        xx = make_dataset(xb)        
        yy = y_batch[j]              
        
        history = model.fit(xx, yy,                            
                  batch_size=len(xx),                
                  epochs=15,                          
                  validation_data=(x_testB, y_testB),  
                  )       
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()             


scores = model.evaluate(x_testA, y_testA, verbose=0)    
print('Test loss:', scores[0])                        
print('Test accuracy:', scores[1])                    			

from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = model.to_json()


with open("model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_num.h5")