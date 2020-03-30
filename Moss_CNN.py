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
# CNN Model
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

# Define input and output of model 
model_final = Model(input=input, output=x)

# Combine the defined model with proper loss, optimizer
# Metric shows the real-time results every batch
model_final.compile(loss = "binary_crossentropy", optimizer = 'SGD', metrics=["binary_accuracy"])

# Monitor validation loss and saves the model when it reaches minimum value
checkpoint = ModelCheckpoint("SOFTMAX_FINAL.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
# Monitor validation accuracy and stops training if the model does not improve for patient number of epochs
early = EarlyStopping(monitor='val_binary_accuracy', min_delta=0, patience=20, verbose=1, mode='max')

# Input using training generator defined above with number of epochs
# Need to call defined checkpoint and early stopping defined above here as callbacks
model_final.fit_generator(generator= training_generator, steps_per_epoch= len(training_generator.filenames) // BATCH_SIZE, 
                          epochs= 100, validation_data= validation_generator, validation_steps=len(validation_generator.filenames) // BATCH_SIZE, 
                          callbacks=[checkpoint,early])
#model_final.save_weights("SOFTMAX_FINAL.h5")


#%%
# Load saved model
model = load_model('SOFTMAX_FINAL.h5')
#model_final.load_weights('SOFTMAX_FINAL.h5')

# Import test images for inputting to the network (requires pre-processing)
gen = ImageDataGenerator(preprocessing_function=preprocess)
iterator = gen.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size), shuffle=False) 

# Import test images for Visualization (does not require pre-processing)
gen_show = ImageDataGenerator()
iterator_show = gen_show.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size), shuffle=False)

# Save imported test images into batch
batch = iterator.next()
batch_show = iterator_show.next()

# Seperate Images and labels
imgs = batch[0]
imgs_show = batch_show[0]
labels = batch[1]

# Define figure window size
ncols, nrows = 8,10
fig = plt.figure( figsize=(ncols*3, nrows*3), dpi=90)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.8, hspace=0.8)
predicted = []

# Print images on the defined figure above
for i, (img_show, img,label) in enumerate(zip(imgs_show, imgs,labels)):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(img_show.astype(np.int))
    preds = model_final.predict(img[np.newaxis,...]) # predict test images
    preds = (np.round(preds*100))
    predicted.append(preds)
    plt.title( '{} \n {}'.format(str(preds), str(label)))
    plt.axis('off')
    
#%%
# Visualize feature maps
# Import VGG-16 model
model = VGG16()

# for i in range(len(model.layers)):
# 	layer = model.layers[i]

# Define input and output for the desired convolution block
feature1 = Model(inputs=model.inputs, outputs=model.layers[1].output)
feature2 = Model(inputs=model.inputs, outputs=model.layers[2].output)

# Extract outputs after specific convolution layer
feature_map1 = feature1.predict(np.expand_dims(img,axis=0))
feature_map2 = feature2.predict(np.expand_dims(img,axis=0))

# Combine multiple feature maps (if necessary)
feature_map = np.concatenate((feature_map1,feature_map2),axis=3)

# Define figure window for feature maps
ix = 1
for _ in range(2):
    for _ in range(5):
        ax = plt.subplot(2, 5, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        idx = random.randint(1,feature_map.shape[3])
        plt.imshow(feature_map[0, :, :, idx],cmap='gray') # Plot feature maps in grayscale
        ix += 1
# show the figure
plt.show()


#%%
# Auto-Encoder part
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# Define training input
training_generator = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode="input",subset='training',
    shuffle=True)

# Define validation input
validation_generator = training_data_generator.flow_from_directory(
    training_dir,
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    class_mode="input",subset='validation',
    shuffle=True)

# bottle-neck size
latent_dim = 32
# Input size
inputs = Input(shape=(224,224,3),name = 'image_input')

##### MODEL 1: ENCODER #####
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

encoder = Model(inputs, z)

##### MODEL 3: DECODER #####
# the bottleneck layer becomes the input to the decoder model
input_z = Input(shape=(latent_dim,))
x_decoded_dense0 = Dense(56 * 56 * 64)(input_z)
x_decoded_activation0 = ELU()(x_decoded_dense0)

x_decoded_reshape0 = Reshape((56, 56, 64))(x_decoded_activation0)
x_decoded_upsample0 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x_decoded_reshape0)
x_decoded_conv0 = Convolution2D(32, (3, 3), padding='same')(x_decoded_upsample0)
x_decoded_activation1 = ELU()(x_decoded_conv0)

x_decoded_upsample1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x_decoded_activation1)
decoded_decoder_img = Convolution2D(1, (3, 3), activation='sigmoid', padding='same')(x_decoded_upsample1)

decoder = Model(input_z, decoded_decoder_img)

# fully defined encoder-decoder model
full = decoder(encoder(inputs))

model = Model(inputs, full)
model.compile(optimizer='adam', loss='mean_squared_error')

checkpoint = ModelCheckpoint("custom_CNN.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='min')
model.fit_generator(generator= training_generator, steps_per_epoch= len(training_generator.filenames) // BATCH_SIZE, 
                          epochs= 100, validation_data= validation_generator, validation_steps=len(validation_generator.filenames) // BATCH_SIZE, 
                          callbacks=[checkpoint,early])
# model.save_weights("custom_CNN.h5")


#%%
# Visualizing clusters using t-SNE
from sklearn import manifold
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]*255.
        img = img.astype(np.uint8).reshape([224,224,3])
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

gen_show = ImageDataGenerator()
iterator_show = gen_show.flow_from_directory(
    test_dir, 
    batch_size=80,
    target_size=(img_size,img_size))


batch = iterator.next()
batch_show = iterator_show.next()

imgs = batch[0]
imgs_show = batch_show[0]
labels = batch[1]

X_encoded = encoder.predict(imgs)
X_decoded = decoder.predict(X_encoded)

# Compute t-SNE embedding of bottleneck
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



