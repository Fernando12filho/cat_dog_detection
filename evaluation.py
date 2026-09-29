import keras
import numpy as np
import tensorflow as tf

DATA_DIR = 'archive/images'
BATCH = 32
SEED = 1

# Load the model first, so the image size comes from it instead of being hardcoded
model = keras.models.load_model('model.keras')
IMG_SIZE = tuple(model.input_shape[1:3])   # e.g. (128, 128)

# Same seed + same split as training = the same 20% the model never trained on
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH,
    label_mode='binary',
)
class_names = val_ds.class_names             # ['cat', 'dog']
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# --- Quick check: accuracy only ---
loss, acc = model.evaluate(val_ds)
print(f'val loss: {loss:.4f}  val accuracy: {acc:.4f}')

# --- Predictions and labels together, in ONE pass (so they stay aligned) ---
all_probs, all_labels = [], []
for images, labels in val_ds:
    probs = model(images, training=False)    # P(dog) for each image in the batch
    all_probs.append(probs.numpy().ravel())
    all_labels.append(labels.numpy().ravel())

probs = np.concatenate(all_probs)
labels = np.concatenate(all_labels).astype(int)
preds = (probs > 0.5).astype(int)            # 0 = cat, 1 = dog

print(f'manual accuracy: {(preds == labels).mean():.4f}')   # should match evaluate()

# --- Confusion matrix: rows = true class, columns = predicted class ---
cm = tf.math.confusion_matrix(labels, preds).numpy()
print(f'{"":>10}{"pred cat":>10}{"pred dog":>10}')
for i, name in enumerate(class_names):
    print(f'{"true " + name:>10}{cm[i, 0]:>10}{cm[i, 1]:>10}')