# Moss Project
#
# Transfer learning from VGG-16 network
# Encoder-Decoder Architecture 
#
#%%
# Importing necessary packages
import matplotlib.pyplot as plt
import cv2
from glob import glob
import os
import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Lambda, GlobalAveragePooling2D
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model


#%%
# Data folder
path = r'C:\Users\admin\Desktop\RGB'

# Within the folder, there should be training and test folders
training_dir = path + r"\training" 
test_dir = path + r"\test" 

# Pre-processing image function (resizing and cropping)
def preprocess(img):
    img_resize = cv2.resize(img, (256, 300), interpolation=cv2.INTER_CUBIC)
    img = img_resize[38:262,16:240,:]
    return img/img.max()

# Extracting training images from defined folder
# Within this function, training and validation dataset is splitted
training_data_generator = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
	height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.3,
    preprocessing_function=preprocess)

# Define image size and batch size
img_size = 224 
BATCH_SIZE = 16

# Generating images for training
training_generator = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode="categorical",subset='training',
    shuffle=True)

# Generating images for validation
validation_generator = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode="categorical",subset='validation',
    shuffle=True)


#%%
from keras.applications.vgg16 import VGG16
from tensorflow.python import keras

os.environ['TF_KERAS'] = '1'
# Importing model and weights of VGG-16, excluding the fully connected layers
vggmodel = VGG16(weights='imagenet', include_top=False)

# Defining input dimension
input = Input(shape=(224,224,3),name = 'image_input')

# Input to imported vgg model
output_vgg16_conv = vggmodel(input)

# Add the fully-connected layers 
x = GlobalAveragePooling2D()(output_vgg16_conv)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(2048, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(4, activation='softmax', name='predictions')(x)

#Create your own model 
model_final = Model(input=input, output=x)

model_final.compile(loss = "binary_crossentropy", optimizer = 'SGD', metrics=["binary_accuracy"])


checkpoint = ModelCheckpoint("vgg16_2.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=20, verbose=1, mode='max')
model_final.fit_generator(generator= training_generator, steps_per_epoch= len(training_generator.filenames) // BATCH_SIZE, 
                          epochs= 100, validation_data= validation_generator, validation_steps=len(validation_generator.filenames) // BATCH_SIZE, 
                          callbacks=[checkpoint,early])
#model_final.save_weights("vgg16.h5")



#%%
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

training_generator = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode="input",subset='training',
    shuffle=True)

validation_generator = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode="input",subset='validation',
    shuffle=True)


latent_dim = 32
inputs = Input(shape=(224,224,3),name = 'image_input')

x = Convolution2D(32, (3, 3), padding='same')(inputs)
x = ELU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Convolution2D(64, (3, 3), padding='same')(x)
x = ELU()(x)
x = MaxPooling2D((2, 2), padding='same')(x)

    # Latent space // bottleneck layer
x = Flatten()(x)
x = Dense(latent_dim)(x)
z = ELU()(x)

    ##### MODEL 1: ENCODER #####
encoder = Model(inputs, z)

    # Create decoder
input_z = Input(shape=(latent_dim,))
x_decoded_dense0 = Dense(56 * 56 * 64)(input_z)
x_decoded_activation0 = ELU()(x_decoded_dense0)

x_decoded_reshape0 = Reshape((56, 56, 64))(x_decoded_activation0)
x_decoded_upsample0 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x_decoded_reshape0)
x_decoded_conv0 = Convolution2D(32, (3, 3), padding='same')(x_decoded_upsample0)
x_decoded_activation1 = ELU()(x_decoded_conv0)

    # Tanh layer
x_decoded_upsample1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x_decoded_activation1)
decoded_decoder_img = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x_decoded_upsample1)

    ##### MODEL 3: DECODER #####
decoder = Model(input_z, decoded_decoder_img)

full = decoder(encoder(inputs))
#autoencoder = Model(input_img, decoded_decoder_img)

model = Model(inputs, full)
model.compile(optimizer='adam', loss='mean_squared_error')

checkpoint = ModelCheckpoint("custom_CNN.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
model.fit_generator(generator= training_generator, steps_per_epoch= len(training_generator.filenames) // BATCH_SIZE, 
                          epochs= 100, validation_data= validation_generator, validation_steps=len(validation_generator.filenames) // BATCH_SIZE, 
                          callbacks=[checkpoint,early])
model.save_weights("custom_CNN.h5")


#%%
from sklearn import manifold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([224,224,3])
#        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    

gen = ImageDataGenerator(preprocessing_function=preprocess)
iterator = gen.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size)) 
#    classes=('Aulacomium palustre','Helodium blandowii','Thuidium recognitum','Tomentypnum nitens'))

gen_show = ImageDataGenerator()
iterator_show = gen_show.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size))
#    classes=('Aulacomium palustre','Helodium blandowii','Thuidium recognitum','Tomentypnum nitens'))
# we can guess that the iterator has a next function, 
# because all python iterators have one. 
batch = iterator.next()
batch_show = iterator_show.next()

imgs = batch[0]
imgs_show = batch_show[0]
labels = batch[1]

X_encoded = encoder.predict(imgs)
X_decoded = decoder.predict(X_encoded)

    # Compute t-SNE embedding of latent space
print("Computing t-SNE embedding...")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
print("Plotting t-SNE visualization...")
fig, ax = plt.subplots()
imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=imgs_show, ax=ax, zoom=0.6)
plt.show()



fig, ax = plt.subplots(1, 2)
ax[0].scatter(X_tsne[:,0],X_tsne[:,1],
	 c=np.argmax(labels, axis=1) ,s=8, cmap='tab10')

#%%
model_final.load_weights('SOFTMAX_FINAL.h5')

gen = ImageDataGenerator(preprocessing_function=preprocess)
iterator = gen.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size), shuffle=False) 
#    classes=('AP','HB','TN'))

gen_show = ImageDataGenerator()
iterator_show = gen_show.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size), shuffle=False)
#    classes=('AP','HB','TN'))
# we can guess that the iterator has a next function, 
# because all python iterators have one. 
batch = iterator.next()
batch_show = iterator_show.next()

imgs = batch[0]
imgs_show = batch_show[0]
labels = batch[1]

ncols, nrows = 8,10
fig = plt.figure( figsize=(ncols*3, nrows*3), dpi=90)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.8, hspace=0.8)
predicted = []

for i, (img_show, img,label) in enumerate(zip(imgs_show, imgs,labels)):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(img_show.astype(np.int))
    preds = model_final.predict(img[np.newaxis,...])
    preds = (np.round(preds*100))
    predicted.append(preds)
#    plt.title( 'Predicted: {} \n Actual: {}'.format(str(preds), str(label)))
    plt.title( '{} \n {}'.format(str(preds), str(label)))

    plt.axis('off')
    
# idx = random.randint(1,32)
# preds = model_final.predict(imgs[idx,np.newaxis,...])
# preds = np.around(preds*100,0)

# plt.title( 'Predicted: {} \n Actual: {}'.format(str(preds), str(labels[idx,:])))
# plt.axis('off')
# plt.imshow(imgs_show[idx,...])

#%%
#image_list = []
#for filename in glob(os.path.join(test_dir, '*.jpg')):
#     im = misc.imread(filename)
#     print('reading', filename)
#     im = cv2.resize(im, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
#     image_list.append(im[np.newaxis,...]/255)
#
# image_list = np.concatenate((image_list), axis=0)

model = VGG16()
# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
# redefine model to output right after the first hidden layer
feature1 = Model(inputs=model.inputs, outputs=model.layers[15].output)
feature2 = Model(inputs=model.inputs, outputs=model.layers[17].output)

feature_map1 = feature1.predict(np.expand_dims(img,axis=0))
feature_map2 = feature2.predict(np.expand_dims(img,axis=0))

feature_map = np.concatenate((feature_map1,feature_map2),axis=3)

# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(2):
    for _ in range(5):
		# specify subplot and turn of axis
        ax = plt.subplot(2, 5, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        idx = random.randint(1,feature_map.shape[3])
		# plot filter channel in grayscale
        plt.imshow(feature_map[0, :, :, idx],cmap='gray')
        ix += 1
# show the figure
plt.show()

