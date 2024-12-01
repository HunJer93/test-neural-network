import tensorflow as tf
# keras is used for image processing
from keras.preprocessing.image import ImageDataGenerator

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
    '/dataset/training_set', # target directory
    target_size= (image_size, image_size), # all images will be resized to 150 x 150 px
    batch_size= batch_size,
    class_mode= class_mode # used for binary_crossentropy loss
)

# PRE-PROCESS THE TEST DATASET
# since this is dataset, DO NOT adjust the image like the train_data_generator since we don't want to edit the images. (only scale adjusted)

test_data_generator = ImageDataGenerator(rescale=1./255)
test_set = test_data_generator.flow_from_directory(
    '/dataset/test_set',
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode=class_mode
)

##################
# END DATA PRE-PROCESSING
##################


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




