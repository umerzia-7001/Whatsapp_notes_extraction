
# importing dependencies
import os
import numpy as np
from glob import glob
from keras.preprocessing.image import *
from model import cnn_model


print('Connect your smartphone to this system, mount your Internal Storage and note the absolute path of WhatsApp folder')
print('For example: "/run/user/1000/gvfs/mtp:host=%5Busb%3A003%2C002%5D/Internal storage/WhatsApp" \n')

WA_path = input('Enter absolute path of WhatsApp picture folder: \n')

WA_img_path = WA_path
WA_img_path.replace('//', '/')
WA_img_path.replace(' ', '/ ')



# define model
model = cnn_model()
# load trained weights , for this you need to train your model
model.load_weights('weights.h5')

notes_path = WA_img_path +'notes'
if not os.path.exists(notes_path):
    os.mkdir(notes_path)

print('Created a notes folder in your WhatsApp Image folder to keep the notes')

def predict(file_path):
    '''
    predict whether file is a notes image
    '''
    img = load_img(file_path, target_size=(124, 124, 3))
    x = img_to_array(img) / 255.
    y = model.predict(np.expand_dims(x, axis=0))
    return np.squeeze(y) > 0.5


# get file paths
files = glob(WA_img_path + '*.*') + glob(WA_img_path + 'Sent/*.*')

# extract notes from WhatsApp Images folder

for count, file_path in enumerate(files):
    if not count % 10: print(str(count) + ' files examined')
    if predict(file_path): # check if the file is one of the notes
        file_name = file_path.split('/')[-1] # get file name
        os.rename(file_path, notes_path + file_name) # move the file to 'notes' folder