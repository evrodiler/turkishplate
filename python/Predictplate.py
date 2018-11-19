import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import Predictplate as p
import os
from os.path import join
import json
import numpy as np
import cv2
import random
import itertools
import re
import datetime
from keras.models import model_from_json
from keras.optimizers import SGD
import shutil
import tensorflow as tf
from keras import backend as K
from collections import Counter
import itertools
#plaka dağılımları 

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
            else:
                fname= filename.split('.')
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
def cnt_plot(df,col,harf,kac):
    df2=pd.DataFrame(df,columns=[col])
    plt.figure(figsize=(16,6)) # this creates a figure 8 inch wide, 4 inch high
    sns.countplot(data=df2,x=col,order=pd.value_counts(df2[col]).iloc[:kac].index)        
     
    plt.xlabel("Plakanın " + harf + ".harfi")
    plt.ylabel("Toplam Sayı")
    plt.title("Plakanın " + harf + " harfine göre Plaka Dağılımı")
    #plt.legend()
    plt.show()

#load model
def load_model(inp_model):
    # load json and create model
     
    json_file = open(inp_model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(inp_model + '.h5')

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # Compile model
    loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    print(inp_model + "model loaded")
    return loaded_model

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
        img_dirpath = dirpath
     
        self.samples = []
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
    
    def next_batch(self,K):
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
                if len(text)<8:    
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