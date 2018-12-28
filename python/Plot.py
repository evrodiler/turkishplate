import pandas as pd
import numpy as np
import os
import seaborn as sns

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


