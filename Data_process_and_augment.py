#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pickle
import os
import shutil
#favorite_color = pickle.load( open( "testlabels", "rb" ))
#favorite_color = pickle.load( open( "testdata", "rb" ))
#print(favorite_color)
#for i in favorite_color.keys():
#    print(favorite_color[i])


# In[43]:


def split_data():
    train_labels = pickle.load( open( "trainlabels", "rb" ))
    train_data = pickle.load( open( "traindata", "rb" ))
    available_files = os.listdir('train')
    for i in train_labels.keys():
        if 'image_'+str(i)+'.jpg' in available_files:
                shutil.copy2('train/image_'+str(i)+'.jpg', 'train_data/'+str(train_labels[i])+'/image_'+str(i)+'.jpg')


# In[44]:


split_data()


# In[39]:


from PIL import Image
def augment_data():
    path = 'train_data/1'
    files = os.listdir(path)
    #print(files)
    for i in files:
        img=Image.open(path+'/'+i)
        img = img.rotate(90)
        img.save(path+'/'+str(90)+i)
        img = img.rotate(180)
        img.save(path+'/'+str(180)+i)
        img = img.rotate(270)
        img.save(path+'/'+str(270)+i)
    return 


# In[40]:


augment_data()


# In[ ]:




