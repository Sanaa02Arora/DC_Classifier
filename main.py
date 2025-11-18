"""
main.py
Run this in PyCharm with Python 3.13.

What it does:
 - Downloads 'cats_vs_dogs' using tensorflow_datasets (first run will download ~ a few 100s MB).
 - Extracts exactly 100 images of each class into ./dataset/dogs/ and ./dataset/cats/
 - Trains a small CNN using ImageDataGenerator on those 200 images (split into train/val)
"""

import os
import shutil
from pathlib import Path
import tensorflow as tf
from PIL import Image
import numpy as np

# ----------  Config ----------
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "dataset"
DOGS_DIR = DATA_DIR / "dogs"
CATS_DIR = DATA_DIR / "cats"
NUM_PER_CLASS = 100   # you said 100 dogs and 100 cats
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 8
# -----------------------------

def prepare_folders():
    # create/clear dataset folders
    for p in (DOGS_DIR, CATS_DIR):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def save_image(np_img, file_path):
    """Save numpy image array to file."""
    img = Image.fromarray(np_img)
    img.save(file_path)

def save_images_from_tfds(num_per_class=100):
    """
    Robust version:
    - prints progress
    - downloads via tfds
    - writes JPEG bytes with tf.io.encode_jpeg to avoid hidden save_img errors
    - stops when it has saved num_per_class for each class
    """
    import tensorflow_datasets as tfds
    # print("Starting tfds load: 'cats_vs_dogs' (this may download the dataset)...")
    # try:
    #     ds, info = tfds.load('cats_vs_dogs', split='train', with_info=True, as_supervised=True)
    # except Exception as e:
    #     print("ERROR: tfds.load failed:", repr(e))
    #     print("Check internet connection, firewall, and that tensorflow-datasets is installed.")
    #     return
    #
    # # Ensure directories exist (in case prepare_folders had issues)
    # CATS_DIR.mkdir(parents=True, exist_ok=True)
    # DOGS_DIR.mkdir(parents=True, exist_ok=True)
    #
    # cat_count = 0
    # dog_count = 0
    # total_seen = 0
    # print("Iterating dataset and saving images...")
    #
    # # convert to numpy iterator safely
    # try:
    #     for i, (image, label) in enumerate(tfds.as_numpy(ds)):
    #         total_seen += 1
    #         if cat_count >= num_per_class and dog_count >= num_per_class:
    #             break
    #
    #         lab = int(label)
    #         # Some tfds images can have different dtypes: convert to uint8 if needed
    #         try:
    #             img_uint8 = tf.image.convert_image_dtype(image, dtype=tf.uint8).numpy()
    #         except Exception as e:
    #             print(f"[{i}] Warning: failed to convert image dtype: {e}. Skipping.")
    #             continue
    #
    #         # Ensure 3 channels (RGB); if grayscale, stack to 3
    #         if img_uint8.ndim == 2:
    #             img_uint8 = np.stack([img_uint8]*3, axis=-1)
    #         elif img_uint8.shape[-1] == 1:
    #             img_uint8 = np.concatenate([img_uint8]*3, axis=-1)
    #
    #         try:
    #             if lab == 0 and cat_count < num_per_class:
    #                 fname = CATS_DIR / f"cat_{cat_count:03d}.jpg"
    #                 # encode jpeg and write bytes (safer/clearer than save_img)
    #                 jpeg = tf.io.encode_jpeg(img_uint8).numpy()
    #                 with open(fname, "wb") as f:
    #                     f.write(jpeg)
    #                 cat_count += 1
    #                 if cat_count % 10 == 0 or cat_count == 1:
    #                     print(f"Saved {cat_count} cats so far (total seen: {total_seen}) -> {fname}")
    #             elif lab == 1 and dog_count < num_per_class:
    #                 fname = DOGS_DIR / f"dog_{dog_count:03d}.jpg"
    #                 jpeg = tf.io.encode_jpeg(img_uint8).numpy()
    #                 with open(fname, "wb") as f:
    #                     f.write(jpeg)
    #                 dog_count += 1
    #                 if dog_count % 10 == 0 or dog_count == 1:
    #                     print(f"Saved {dog_count} dogs so far (total seen: {total_seen}) -> {fname}")
    #
    #         except Exception as e:
    #             print(f"[{i}] ERROR while saving file: {e}. Continuing.")
    #             continue
    #
    # except Exception as e:
    #     print("ERROR iterating tfds dataset:", repr(e))
    #
    # print(f"Finished. Saved cats: {cat_count}, dogs: {dog_count}, total seen: {total_seen}")
    # print("Dataset dir:", DATA_DIR.resolve())

    # Load dataset
    ds, info = tfds.load("cats_vs_dogs", split="train", with_info=True)
    print("Dataset loaded.")

    dog_count = 0
    cat_count = 0
    max_per_class = 100

    for example in ds:
        img = example["image"].numpy()
        label = example["label"].numpy()  # 0 = cat, 1 = dog

        if label == 0 and cat_count < max_per_class:
            save_image(img, f"./dataset/cats/cat_{cat_count + 1}.jpg")
            cat_count += 1

        elif label == 1 and dog_count < max_per_class:
            save_image(img, f"./dataset/dogs/dog_{dog_count + 1}.jpg")
            dog_count += 1

        print(f"Saved: cats={cat_count}, dogs={dog_count}", end="\r")

        if cat_count >= max_per_class and dog_count >= max_per_class:
            break

    print("\nDone! Images saved into ./dataset/")


def build_and_train():
    # Data generators
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # 80% train, 20% val from the folders
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Simple CNN
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    # Save model
    model.save(PROJECT_DIR / "dog_cat_classifier.h5")
    print("Model saved to dog_cat_classifier.h5")

    # Optional: show training history (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend(); plt.title('Accuracy')
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(); plt.title('Loss')
        plt.show()
    except Exception as e:
        print("Matplotlib not available or failed to plot:", e)

def main():
    # # 1) prepare folders
    # prepare_folders()

    # 2) download & save images (100 per class)
    # save_images_from_tfds(NUM_PER_CLASS)

    # 3) build and train
    build_and_train()

if __name__ == "__main__":
    main()
