#augmentation procedures
import os
import random
from random import randint
import PIL
import numpy as np

# list create for numbers or letters
def create_list(data_dir,harf):    
    
    lst = []
    # Append empty lists in first two indexes.
    for i in range(len(os.listdir(data_dir))):
        lst.append([])
        
    label=0
    i=0
    for name in os.listdir(data_dir):  
        lst[i].append(name)
        
        if harf:
            spl= name.split('.')        
            lst[i].append(spl[0][0].upper())
        else:
            if i%10==0 and i!=0:
                label=label+1 
            lst[i].append(label)  
            
        i=i+1    
        
    return lst
#####################################################################
#random letter or digit generator
def rnd_list(h,kac_tane,data_dir,frm,too):
    label=''
    lst=[]
    s = [randint(frm, too-1) for p in range(0, kac_tane)]
       
    for i in range(len(s)):       
        label = label + str(h[s[i]][1])
        lst.append(data_dir + '/' + h[s[i]][0])
    
    #if label is 0,00 or 000 etc recall the proc
    if RepresentsInt(label):
        #print('2 label in proc:',label)
        if int(label)==0 and kac_tane!=1: #only for the last part of plate
            #print('3 label in proc:',label)
            rnd_list(h,kac_tane,data_dir,frm,too) 
        
    return label,lst


#image concatenate and save pic
def img_concat(lst, dirname, filename):
    imgs = [ PIL.Image.open(i).convert('RGB') for i in lst ]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    
    try:
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    except:
        print("hstack hata", imgs)
        
    # save that beautiful picture
    try:
        imgs_comb = PIL.Image.fromarray(imgs_comb)
    except:
        print("PIL.Image.fromarray hata")
        
    try:
        imgs_comb.save(dirname + filename )    
        print(filename , "image saved in folder", dirname)
    except:       
        print(filename , "image not saved due to io error")

        
######################################################################  
#file rename
def f_rename():
    inp_img_folder= DATA_DIR + '/plaka_default/rakamlar/'
    out_img_folder= DATA_DIR + "/plaka_default/out/"
    import shutil
    cnt=0
    source_file=''
    dest_file = ''
    for name in os.listdir(data_dir):    
        try:
            #if cnt<2:
                #print(index)
                source_file = inp_img_folder +  name                      

                s = name.split('.')
                dest_file = out_img_folder + str(int(s[0]) +100) + ".png"
                print("source_file:" + str(source_file) , ",dest_file:" + str(dest_file))
                shutil.copy(source_file,dest_file)        
                cnt+= 1    
                #print(cnt)
            #else:
             #   break    
        except Exception as e:
            print(str(e))
            
#img to convert to RGB from RGBA
def conv_RGB(dir_name, new_dirname):
    for i in os.listdir(dir_name):    
        img = PIL.Image.open(dir_name + i)
        img =img.convert('RGB')
        img.save(new_dirname +  i)
#                 
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False        

def add_tr(data_dir,lstd0 , lstd1 , lst , lstd2):
    # tr picture added to first piece
    tr = data_dir + '/plaka_default/tr.PNG' #Default tr pic
    lst_merge=[]
    lst_merge.append(tr)
    lst_merge= lst_merge+ lstd0 + lstd1 + lst +lstd2
    
    return lst_merge

    