import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
import json
import numpy as np
import cv2
import random
import itertools
import re
import datetime
import shutil
import tensorflow as tf
from collections import Counter
import itertools
#KERAS
from keras.models import model_from_json
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model, load_model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import cv2
from keras.layers import Dropout


#plaka tanıma ocr 
letters_train = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z','X','W','Q']
letters = sorted(list(letters_train))

    
def check_word(word):
    #spec_symbols = '.abcdefghijklmnopqrstuvwyz'
    spec_symbols = 'c.a'
    match = [l in spec_symbols for l in word]       
    return sum(match) > 0 

def check_plate(word):
    spec_symbols = '#'
    match = [l in spec_symbols for l in word]       
    return sum(match) >= 2 #başında ve devamında iki tane özel char varsa true

class TextImageGenerator:
    
    def __init__(self, 
                 dirpath,
                 tag,
                 img_w, img_h, 
                 batch_size, 
                 downsample_factor,
                 max_text_len=11):
        
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        
        #evrim       
        img_dirpath = dirpath + '/' + tag  #test,dev or train      
        start='#'
        end='#'
        self.samples = []
        
        for filename in os.listdir(img_dirpath): 
            name, ext = os.path.splitext(filename)            
            #print("name:", name," ,name[0:11]:",name[0:11]) #," ,ext:",ext)
            
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)                             
                if check_plate(name): # if plate 29.may
                   description= (name.split(start))[1].split(end)[0]  
                   if len(description)<7 or len(description)>8:
                      print('hata! ' , filename + 'nolu plakanın etiketi' + str(len(description)) + 'uzunlukta!' )
                   else:
                      self.samples.append([img_filepath, description])
                      #print("plate description:", description)
                else:    
                    if check_word(name[0:11]):
                        print("init error!!:", name[0:11])
                    else:                          
                        self.samples.append([img_filepath, name[0:11]])   #evrim     
                        #print("container:", name[0:11])
             
            self.n = len(self.samples)
            self.indexes = list(range(self.n))
            self.cur_index = 0   
        
        
        
    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            if check_word(text):               
                print("build data error:",text)
            else:
                self.texts.append(text)
        
    def get_output_size(self):
        return len(letters) + 1
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):
      
        max_text_len= 11
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                if check_word(text):
                    print("\n in next batch wrong container number:", text)                    
                else:    
                    X_data[i] = img
                    
                    #evrim starts
                    if len(text) < max_text_len:    
                       temp= text_to_labels(text)   
                       for xx in range(max_text_len-len(temp)):
                           temp.append(len(letters) + 1)     
                        
                       Y_data[i] = temp
                    else:
                       Y_data[i] = text_to_labels(text)
                    #evrim ends
                    
                    #print("Y_data:",len(Y_data[i] ), '--', Y_data[i] )
                    
                    
                    #Y_data[i] = text_to_labels(text)
                    
                    source_str.append(text)
                    label_length[i] = len(text)   
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
            
# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

####################################
def load_model(fmodel,lrate):
    # load json and create model    
    json_file = open(fmodel + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(fmodel + '.h5')

    sgd = SGD(lr=lrate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # Compile model
    loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    return loaded_model
    
    
############################################
def predict_single_plate(sess,loaded_model,img,img_w,img_h): 
    #img_w,img_h = 128, 64
   
    #img = cv2.imread(img_filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32)
    img /= 255


    net_inp = loaded_model.get_layer(name='the_input').input
    net_out = loaded_model.get_layer(name='softmax').output

    if K.image_data_format() == 'channels_first':
        X_data = np.ones([1, 1, img_w, img_h])
    else:
        X_data = np.ones([1, img_w, img_h, 1])
    
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    X_data[0] = img

    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value)
#   sess.close()
    return pred_texts


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#evrim  starts
"""
#original
def labels_to_text(labels):    
    return ''.join(list(map(lambda x: letters[int(x)], labels)))
"""
def labels_to_text(labels):
    #evrim starts
    if labels[7]== len(letters) + 1:
        labels= labels[:7]
    #evrim ends
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

#evrim ends

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

def train2gru(img_w,img_h,path_data,load,path_name_model,epoch_cnt,lrate,ibatchsize):
    # Input Parameters

    #ORiginal!!!
    
    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    batch_size = ibatchsize
    downsample_factor = pool_size ** 2
   
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
    
    tiger_train = TextImageGenerator(path_data, 'train', img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()
    tiger_val = TextImageGenerator(path_data, 'val', img_w, img_h, batch_size, downsample_factor)
    tiger_val.build_data()
   
    
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    #evrim
    inner = Dropout(0.2)(inner)    
    #evrim ends
    
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    
     #evrim
    inner = Dropout(0.2)(inner)    
    #evrim ends
    
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    
    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
    
    #evrim
    inner = Dropout(0.2)(inner)    
    #evrim ends

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
   

    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    
    
    #evrim
    """
    gru2_merged = add([gru_2, gru_2b]) 
    gru_3 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru3')(gru2_merged)
    gru_3b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru3_b')(gru2_merged)
    """   
    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
                  #name='dense3')(concatenate([gru_3, gru_3b]))     
                  name='dense2')(concatenate([gru_2, gru_2b]))     
    
    
    y_pred = Activation('softmax', name='softmax')(inner)
    
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=lrate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5) #evrim    
    
    if load:
        model = load_model('./tmp_model.h5', compile=False)
    else:
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)    
   
    
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
                  optimizer=sgd,
                  metrics=['accuracy'])
  
    if not load:
        # captures output of softmax so we can decode the output during visualization
        test_func = K.function([input_data], [y_pred])
       
        
        history= model.fit_generator(generator=tiger_train.next_batch(), 
                            steps_per_epoch=tiger_train.n,
                            epochs=epoch_cnt, 
                            validation_data=tiger_val.next_batch(), 
                            validation_steps=tiger_val.n)
    
    #save model**************
  
    # serialize model to JSON
    model_json = model.to_json()
    with open(path_name_model + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(path_name_model + ".h5")
    
    print("Saved" + path_name_model + " model to disk") 
       
    
    
    #model.save(path_name_model + ".h5")  bu çalışmadı
    #RuntimeError: Unable to create attribute (object header message is too large)

    
    return model


