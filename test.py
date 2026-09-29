import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt

DATA_DIR = 'archive/images'
IMG_SIZE = (128, 128)
BATCH = 32
SEED = 1

# --- Data ---
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='both',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='binary',
)
print(train_ds.class_names)  # ['cat', 'dog']

# Sanity checks before spending time on training
images, labels = next(iter(train_ds))
assert images.shape[1:] == (*IMG_SIZE, 3), images.shape
assert set(labels.numpy().ravel()) <= {0.0, 1.0}

AUTOTUNE = tf.data.AUTOTUNE
to_uint8 = lambda x, y: (tf.cast(x, tf.uint8), y)

train_ds = train_ds.take(150)
# Load the next batch while the current one trains
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# --- Model ---
model = keras.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Rescaling(1./255),                    # 0-255 -> 0-1

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),             # much lighter than Flatten
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),       # P(dog)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train ---
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
history = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[early_stop])

# --- Save Model ---
model.save('model.keras',overwrite=True)

# --- Look at what happened ---
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
plt.show()