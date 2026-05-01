# =========================================================
# INSTALL
# =========================================================
#!pip install -q librosa

# =========================================================
# IMPORTS
# =========================================================
import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob

# =========================================================
# PARAMETERS (OPTIMIZED)
# =========================================================
dataset_path = "/kaggle/input/datasets/prasenjitbhakat007/audio-updated/Audio"

SR = 16000                # reduced
DURATION = 1.0
SAMPLES = int(SR * DURATION)

N_MELS = 32              # reduced
FFT = 1024
HOP = 512

BATCH_SIZE = 32
EPOCHS = 30

# =========================================================
# LOAD FILES
# =========================================================
class_names = sorted(os.listdir(dataset_path))
class_map = {name:i for i,name in enumerate(class_names)}
NO_CLASSES = len(class_names)

train_files = []
val_files = []

np.random.seed(42)

for c in class_names:
    files = glob(os.path.join(dataset_path,c,"*.wav"))
    np.random.shuffle(files)
    split = int(0.8*len(files))
    train_files += files[:split]
    val_files += files[split:]

print("Classes:",class_map)

# =========================================================
# FEATURE EXTRACTION (OPTIMIZED)
# =========================================================
def extract_logmel(file):

    audio,_ = librosa.load(file,sr=SR)

    if len(audio) < SAMPLES:
        audio = np.pad(audio,(0,SAMPLES-len(audio)))
    else:
        audio = audio[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel)

    #  better normalization
    logmel = (logmel - np.mean(logmel)) / (np.std(logmel)+1e-6)

    return logmel.astype(np.float32)

# =========================================================
# DATA GENERATOR
# =========================================================
def generator(file_list):

    while True:
        np.random.shuffle(file_list)

        for file in file_list:
            x = extract_logmel(file)

            label = os.path.basename(os.path.dirname(file))
            y = class_map[label]

            x = np.expand_dims(x,-1)
            yield x, np.int32(y)

# =========================================================
# DATASET
# =========================================================
spec_shape = extract_logmel(train_files[0]).shape

train_dataset = tf.data.Dataset.from_generator(
    lambda: generator(train_files),
    output_signature=(
        tf.TensorSpec(shape=(spec_shape[0],spec_shape[1],1),dtype=tf.float32),
        tf.TensorSpec(shape=(),dtype=tf.int32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: generator(val_files),
    output_signature=(
        tf.TensorSpec(shape=(spec_shape[0],spec_shape[1],1),dtype=tf.float32),
        tf.TensorSpec(shape=(),dtype=tf.int32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =========================================================
# MODEL (TINYML OPTIMIZED)
# =========================================================
def tiny_model(input_shape):

    inputs = tf.keras.Input(shape=input_shape)

    #  Depthwise Separable Conv (very important)
    x = tf.keras.layers.SeparableConv2D(8,(3,3),padding="same",activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.SeparableConv2D(16,(3,3),padding="same",activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.SeparableConv2D(24,(3,3),padding="same",activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(32, activation="relu")(x)   # small dense
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(NO_CLASSES,activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

model = tiny_model((spec_shape[0],spec_shape[1],1))
model.summary()

# =========================================================
# COMPILE
# =========================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# CALLBACKS
# =========================================================
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
]

# =========================================================
# TRAIN
# =========================================================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=400,     # reduced
    validation_steps=100,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================================================
# SAVE MODEL
# =========================================================
model.save("tiny_audio_model.keras")

# =========================================================
# TFLITE INT8 CONVERSION
# =========================================================
def representative_dataset():
    gen = generator(train_files)
    for _ in range(100):
        x,_ = next(gen)
        yield [np.expand_dims(x,0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.representative_dataset=representative_dataset
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type=tf.int8
converter.inference_output_type=tf.int8

tflite_model = converter.convert()

with open("TinyAudio_INT8.tflite","wb") as f:
    f.write(tflite_model)

print("Model size:", os.path.getsize("TinyAudio_INT8.tflite")/(1024*1024), "MB")

# =========================================================
# TEST
# =========================================================
interpreter = tf.lite.Interpreter(model_path="TinyAudio_INT8.tflite")
interpreter.allocate_tensors()

print(" Ready for ESP32 (no PSRAM)")