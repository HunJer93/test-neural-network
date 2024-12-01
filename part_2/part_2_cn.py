import tensorflow as tf
# keras is used for image processing
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

####################
# PRE-PROCESSING THE DATASET
####################

# augmentation configuration for trainging the dataset

# more info here: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# rescale handles feature scaling, and the values are passed to prevent overfitting on the training set
# shear is for randomly applying shearing transformations: https://en.wikipedia.org/wiki/Shear_mapping
# zoom range is for randomly zooming inside of pictures
# the horizontal flip is for randomly flipping images (needed for real-world images where there is no assumed horizontal assymetry)
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# PRE-PROCESS THE TRAINING SET

# variables for training set generator instead of hard coded values
image_size = 64
batch_size = 32
class_mode = 'binary' # could be binary or categorical (yes/no vs. multiple outcomes)

# this generator reads the pictures found in '/dataset/training_set', and generates batches to train our network.
training_set = train_data_generator.flow_from_directory(
    'dataset/training_set', # target directory
    target_size= (image_size, image_size), # all images will be resized to 150 x 150 px
    batch_size= batch_size,
    class_mode= class_mode # used for binary_crossentropy loss
)

# PRE-PROCESS THE TEST DATASET
# since this is dataset, DO NOT adjust the image like the train_data_generator since we don't want to edit the images. (only scale adjusted)

test_data_generator = ImageDataGenerator(rescale=1./255)
test_set = test_data_generator.flow_from_directory(
    'dataset/test_set',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode=class_mode
)


##################
# BUILD THE CNN
##################

cnn = tf.keras.models.Sequential()

# create the convolutional layer

cnn.add(tf.keras.layers.Conv2D(
    filters=32, # 32 is common filter size
    kernel_size=3, # 3x3 filter
    activation='relu', # using rectifier activation function
    input_shape=[image_size, image_size, 3 ] # dimension size of the images (used same dimensions from above) and 3 (r, g, b) for color images (we would use 1 if the images were black and white) (only needed on first layer)
))

# create pooling layer
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2, # when creating our pool size, we are moving 2x2 frames (pixles)
    strides=2 # moving 2x2 frame (pool_size) 2 pixels at a time
))

# create 2nd convolutional layer

cnn.add(tf.keras.layers.Conv2D(
    filters=32, # 32 is common filter size
    kernel_size=3, # 3x3 filter
    activation='relu', # using rectifier activation function
))

# create 2nd pooling layer
cnn.add(tf.keras.layers.MaxPool2D(
    pool_size=2, # when creating our pool size, we are moving 2x2 frames (pixles)
    strides=2 # moving 2x2 frame (pool_size) 2 pixels at a time
))


# add flattening layer

cnn.add(tf.keras.layers.Flatten())


# connect the layers together

cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) # adding larger number of neurons for accuracy and using rectifier function

# create the output layer

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # using sigmoid for binary classification (multiclass would use softmax activation)


##################
# TRAIN THE CNN
##################

# compile the CNN
cnn.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['accuracy']
)

# train the CNN on the training set and use test set to validate
cnn.fit(
    x = training_set,
    validation_data= test_set,
    epochs=25 # number of times the network runs.
)

##################
# USING CNN TO MAKE A PREDICTION
##################

# for giggles, request user input for image 
from pathlib import Path
user_input = ''


while user_input != 'q':
    print('\nThe CNN has been trained! \n')
    print('Enter the file path to see if the image is a cat or a dog')
    print('You can also enter Q to quit\n')
    
    user_input = input('Enter the file name here from the single_prediction directory (don\'t include the .jpg extension): ').lower()
    
    if user_input != 'q':
        file_path = 'dataset/single_prediction/' + user_input + '.jpg'
        print('searching for file ' + user_input + '.jpg ' + ' in ' + file_path + '\n')
        if Path(file_path).is_file():
            # load image from the single_prediction directory
            # resize using target_size used to make sure the images are the same size of the training set
            test_image = image.load_img(file_path, target_size= (image_size, image_size))

            # convert image into an array to compare 1:1 against CNN output
            test_image = image.img_to_array(test_image)

            # add extra dimention to make up for batch size (single image vs 32 images in training) so that the test_image can be input into the cnn model
            test_image = np.expand_dims(test_image, axis = 0)

            # output the result
            result = cnn.predict(test_image)

            # get indices from the training set (is 1/0 cat or dog?)
            training_set.class_indices
            if result[0][0] == 1: # use 0 index to access first batch (the only batch) and index 0 to access the prediction
                prediction = 'It looks like a Dog'
            else:
                prediction = 'It looks like a Cat'
                
            print(prediction)
        else:
            print('.jpg file was not found at ' + file_path)
    
    
print('Exiting now. Goodbye!')