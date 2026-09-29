import keras
from keras import layers
import tensorflow as tf

DATA_DIR = 'archive/images'
IMG_SIZE = (128, 128)
BATCH = 32
SEED = 1

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='binary',
) 

AUTOTUNE = tf.data.AUTOTUNE
to_uint8 = lambda x, y: (tf.cast(x, tf.uint8), y)

val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

model = keras.models.load_model('model.keras')
model.evaluate(val_ds)
classes = model.predict(val_ds, batch_size=10)

