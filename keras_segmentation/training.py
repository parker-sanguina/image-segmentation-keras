from models.unet import unet_mini
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from hands import HandConfig, HandDataset, DataController

config = HandConfig()

HAND_DIR = "/Users/parker/Desktop/deep-learning/DeepLearning-Data/egohands_data"

print("Loading training data...")

# Training dataset.
training = HandDataset()
training.load_hands(HAND_DIR, "training")
training.prepare()

print("Loading validation data...")

# Validation dataset
validation = HandDataset()
validation.load_hands(HAND_DIR, "validation")
validation.prepare()

# Define data generators
train_generator = DataController(training, config, shuffle=True, augmentation=None).generate_data()
val_generator = DataController(validation, config, shuffle=True, augmentation=None).generate_data()

model = unet_mini(4, 500, 500)

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-3), loss=BinaryCrossentropy())

model.fit(train_generator, epochs=10, steps_per_epoch=100, validation_data=val_generator, validation_steps=10, workers=16, use_multiprocessing=True, verbose=1)