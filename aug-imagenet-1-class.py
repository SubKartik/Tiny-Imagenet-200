import tensorflow
from tensorflow import keras
import numpy as np
import pandas as pd
import os

scale=25
ngen=1
batchsize=65

savedir='/data/images'
imdir='n02124075'
dpath='/data/'+imdir
flist=os.listdir(dpath)
cdf=pd.DataFrame()
cdf['filename']=flist
cdf['class']=imdir

datagen=keras.preprocessing.image.ImageDataGenerator(
#        rescale=1./255,
#        horizontal_flip=True,
#        rotation_range=20,
#        height_shift_range=0.2,
#        width_shift_range=0.2,
#        vertical_flip=True
    )

wres=128*scale
hres=128*scale

#if os.path.isdir(savedir) == False: 
#    print(os.path.isdir(savedir))
#    os.makedirs(savedir)

i=0

for xb,yb in datagen.flow_from_dataframe(cdf,
        directory=dpath,
        x_col="filename",
        y_col="class",
        batch_size=batchsize,
        class_mode='binary',
        save_to_dir=savedir,
        save_format='JPEG',
        save_prefix='aug',
        target_size=(hres,wres)):

    i=i+1
    print('Outer:',i)
    if i == ngen*20:
        print('Inner:',i)
        break

