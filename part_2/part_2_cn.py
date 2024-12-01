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

