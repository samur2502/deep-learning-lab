import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob

from helper_functions import get_confusion_matrix, plot_confusion_matrix, plot_loss_accuracy_curves

# Training configs
epochs = 3
batch_size = 32

# Paths for dataset
train_path = 'datasets/fruits-360-100x100-small/Training'
test_path = 'datasets/fruits-360-100x100-small/Test'

# Number of classes
folders = glob(train_path + '/*')

# Load VGG16 model pre-trained on ImageNet, exclude top layer
VGG = VGG16(input_shape=[100, 100, 3], weights='imagenet', include_top=False)

# Freeze pre-trained layers
for layer in VGG.layers:
    layer.trainable = False

# Add new classification layers
x = Flatten()(VGG.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Create model
model = Model(inputs=VGG.input, outputs=prediction)

# Compile model with categorical crossentropy and RMSprop optimizer
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)

# Image augmentation for training and validation
train_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

# Train and validation data generators
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  
)

valid_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(100, 100),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Test data generator (no augmentation)
test_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    test_path, 
    target_size=(100, 100), 
    batch_size=1,
    class_mode='categorical'
)

# Train model
r = model.fit(
    train_generator,
    validation_data=valid_generator,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1
)

# Plot loss and accuracy curves
plot_loss_accuracy_curves(r.history)

# Confusion matrix on test data
test_cm, y_true, y_pred = get_confusion_matrix(model, test_generator, test_generator.samples)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=list(test_generator.class_indices.keys()))
