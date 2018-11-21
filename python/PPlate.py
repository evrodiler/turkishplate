import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import os
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

#plaka tanıma ocr 
letters_train = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z']
letters = sorted(list(letters_train))

#from directory taking all plate numbers
def datacreate(path,path_true,strx):   
    data=[]
    if path_true:
        start='#'
        end='#'
        for filename in os.listdir(path): 
            if filename.find('#')!=-1: #special char found            
                fname= (filename.split(start))[1].split(end)[0]             
                #print(fname)
            else:
                fname= filename.split('.')[0]
                #print(fname)
            if (len(fname[0])< 9):    
                data.append(list(fname))
            else:
                print("fname gt 8:",fname)
    else:
        for stri in strx:
            data.append(list(stri))      
        
    df= pd.DataFrame(data,columns=['1','2','3','4','5','6','7','8'])      
    
    return df


#plotting data
def cnt_plot(df,col,harf,kac,msg):
    df2=pd.DataFrame(df,columns=[col])
    plt.figure(figsize=(16,6)) # this creates a figure 8 inch wide, 4 inch high
    sns.countplot(data=df2,x=col,order=pd.value_counts(df2[col]).iloc[:kac].index)        
     
    plt.xlabel("Plakanın " + harf + ".harfi")
    plt.ylabel("Toplam Sayı")
    plt.title("Plakanın " + harf + " harfine göre " + msg + " Plaka Dağılımı")
    #plt.legend()
    plt.show()


#classify plates as matched or non_matched 
def all_plates_classify(sess,tiger_test,net_inp,net_out):
    nmatch_str=[]
    match_str=[]
    matched=0
    non_matched=0
    i=0
    for inp_value, _ in tiger_test.next_batch(K):
        testset_cnt = inp_value['the_input'].shape[0]
        X_data = inp_value['the_input']
       
        net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
        
        pred_texts = decode_batch(net_out_value)

        labels = inp_value['the_labels']
        texts = []     
        
        for label in labels:        
            text = labels_to_text(label)        
            texts.append(text)  
            #print("evrim2")
        for cnt in range(testset_cnt-1):                    
            if pred_texts[cnt] == texts[cnt]:
                matched += 1            
                match_str.append(texts[cnt])   
            else:
                non_matched +=  1
                nmatch_str.append(texts[cnt]) 
            #print("cnt:", cnt)   
        return nmatch_str,match_str,non_matched,matched
    
######################################        
#               train model    
#####################################    
def train(img_w,img_h,path_data,load,path_name_model,epoch_cnt):
    # Input Parameters

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)
        
    batch_size = 32
    downsample_factor = pool_size ** 2
   
    tiger_train = TextImageGenerator(path_data, 'train', img_w, img_h, batch_size, downsample_factor)
    tiger_train.build_data()
    tiger_val = TextImageGenerator(path_data, 'val', img_w, img_h, batch_size, downsample_factor)
    tiger_val.build_data()
   
    
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    
    
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    
    # transforms RNN output to character activations:
    inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',
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
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    
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
    
    return model
####################################
def load_model(fmodel):
    # load json and create model    
    json_file = open(fmodel + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(fmodel + '.h5')

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # Compile model
    loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    return loaded_model
    
    
    
def check_word(word):
    spec_symbols = '#'
    match = [l in spec_symbols for l in word]       
    return sum(match) >= 2 #başında ve devamında iki tane özel char varsa true

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def labels_to_text(labels):
    #evrim starts
    if labels[7]== len(letters) + 1:
        labels= labels[:7]
    #evrim ends
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

class TextImageGenerator:
    
    def __init__(self, 
                 dirpath,
                 tag,
                 img_w, img_h, 
                 batch_size, 
                 downsample_factor,
                 max_text_len=8):
        
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
            
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)
                if check_word(name):                
                    description= (name.split(start))[1].split(end)[0] 
                else:
                    description = name     
                #print("desc:",description)
                if len(description)<7 or len(description)>8:
                    print('hata! ' , filename + 'nolu plakanın etiketi' + str(len(description)) + 'uzunlukta!' )
                else:
                    self.samples.append([img_filepath, description])
        
        """        
        #evrim
        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        start='#'
        end='#'
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)
                if check_word(name):                
                    description= (name.split(start))[1].split(end)[0] 
                else:
                    description = name     
                #print("desc:",description)
                self.samples.append([img_filepath, description])
        
        img_dirpath = join(dirpath, 'img')
        ann_dirpath = join(dirpath, 'ann')
        self.samples = []
        for filename in os.listdir(img_dirpath):
            name, ext = os.path.splitext(filename)
            if ext in ['.png', '.jpg']:
                img_filepath = join(img_dirpath, filename)
                json_filepath = join(ann_dirpath, name + '.json')
                ann = json.load(open(json_filepath, 'r'))
                description = ann['description']
                tags = ann['tags']
                if tag not in tags:
                    continue
                if is_valid_str(description):
                    self.samples.append([img_filepath, description])
        """
        
        
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
        max_plate_len= 8
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
                X_data[i] = img
                #print("text:",text)
                #evrim starts
                if len(text) < max_plate_len:    
                    temp= text_to_labels(text)                    
                    temp.append(len(letters) + 1)     
                    Y_data[i] = temp
                else:
                    Y_data[i] = text_to_labels(text)
                #evrim ends
                
                source_str.append(text)
                label_length[i] = len(text)
                #label_length[i] = 8  #evrim
                
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

#prints the total num of test dev and train examples
def Print_overall(ann_dirpath,test_dirpath):

    test_cnt= len(os.listdir(test_dirpath))
    val_cnt=0
    train_cnt=0
    for fx in os.listdir(ann_dirpath):
        with open(ann_dirpath + "/" + fx) as f:       
            source_data = json.load(f) 
            #print(source_data['tags'])
            if source_data['tags'] == ['val']:
                val_cnt+=1
            else:
                train_cnt +=1   
                
    print("Total count of files:",len(os.listdir(ann_dirpath))+ len(os.listdir(test_dirpath)))
    print("Total count of training set:",train_cnt)
    print("Total count of val set:",val_cnt)    
    print("Total count of test set:",test_cnt)
    return test_cnt