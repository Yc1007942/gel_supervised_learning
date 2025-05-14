# import os
# import glob
# import json
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt

# def load_flow_data(material_pattern="material_*", cycle_pattern="cycle_*"):
#     flows = []
#     labels = []
#     for mat_dir in sorted(glob.glob(material_pattern)):
#         # zero-based class index
#         mat_id = int(os.path.basename(mat_dir).split("_")[-1]) - 1
#         for cycle_dir in sorted(glob.glob(os.path.join(mat_dir, cycle_pattern))):
#             path = os.path.join(cycle_dir, "flow.json")
#             if not os.path.isfile(path):
#                 continue
#             Ox, Oy, Cx, Cy, _ = json.load(open(path))
#             dx = np.array(Cx) - np.array(Ox)
#             dy = np.array(Cy) - np.array(Oy)
#             # reshape into 6x8 grid; adjust if your grid is different
#             dx = dx.reshape((6, 8))
#             dy = dy.reshape((6, 8))
#             flows.append(np.stack([dx, dy], axis=-1))
#             labels.append(mat_id)
#     return np.array(flows, dtype=np.float32), np.array(labels, dtype=np.int32)

# def build_cnn(input_shape, num_classes):
#     inp = tf.keras.Input(shape=input_shape)
#     x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
#     x = tf.keras.layers.MaxPool2D()(x)
#     x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(64, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
#     model = tf.keras.Model(inp, out)
#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

# if __name__ == "__main__":
#     # 1) Load data
#     X, y = load_flow_data()
#     num_classes = len(np.unique(y))
#     y_cat = to_categorical(y, num_classes=num_classes)

#     # 2) Train/test split
#     X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
#         X, y_cat, y, test_size=0.3, stratify=y, random_state=42
#     )

#     # 3) Build & summarize model
#     model = build_cnn(input_shape=X_train.shape[1:], num_classes=num_classes)
#     model.summary()
#     # after training and before or after evaluation:
#     model.save("material_cnn.h5")
#     print("Saved model to material_cnn.h5")


#     # 4) Train with EarlyStopping
#     es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
#     history = model.fit(
#         X_train, y_train,
#         validation_split=0.1,
#         epochs=50,
#         batch_size=32,
#         callbacks=[es]
#     )

#     # 5) Evaluate on test set
#     test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
#     print(f"\nTest accuracy: {test_acc * 100:.2f}%\n")

#     # 6) Classification report
#     y_pred_probs = model.predict(X_test)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     print("Classification Report:")
#     print(classification_report(y_test_labels, y_pred))

#     # 7) Confusion matrix plot
#     cm = confusion_matrix(y_test_labels, y_pred)
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest')
#     classes = sorted(np.unique(y))
#     plt.xticks(range(len(classes)), classes)
#     plt.yticks(range(len(classes)), classes)
#     plt.xlabel('Predicted Class')
#     plt.ylabel('True Class')
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()

#     # 8) Training curves: accuracy and loss
#     # Accuracy
#     plt.figure()
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.legend(['train', 'val'])
#     plt.tight_layout()
#     plt.show()

#     # Loss
#     plt.figure()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend(['train', 'val'])
#     plt.tight_layout()
#     plt.show()









#!/usr/bin/env python3
import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPool2D, GlobalAveragePooling2D, Dense, Dropout, Normalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_flow_data(material_pattern="material_*", cycle_pattern="cycle_*"):
    flows, labels = [], []
    for mat_dir in sorted(glob.glob(material_pattern)):
        if not os.path.isdir(mat_dir):
            continue
        parts = os.path.basename(mat_dir).split("_")
        if not parts[-1].isdigit():
            continue
        mat_idx = int(parts[-1]) - 1
        for cycle_dir in sorted(glob.glob(os.path.join(mat_dir, cycle_pattern))):
            if not os.path.isdir(cycle_dir):
                continue
            fpath = os.path.join(cycle_dir, "flow.json")
            if not os.path.isfile(fpath):
                continue
            Ox, Oy, Cx, Cy, _ = json.load(open(fpath))
            dx = np.array(Cx) - np.array(Ox)
            dy = np.array(Cy) - np.array(Oy)
            dx = dx.reshape((6, 8))  # reshape to your grid dims
            dy = dy.reshape((6, 8))
            flows.append(np.stack([dx, dy], axis=-1))
            labels.append(mat_idx)
    return np.array(flows, dtype=np.float32), np.array(labels, dtype=np.int32)


def build_dataset(X, y, batch_size=32, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if augment:
        def augment_fn(x, y):
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
            x = x + noise
            x = tf.image.random_flip_left_right(x)
            return x, y
        ds = ds.shuffle(len(X)).map(augment_fn, tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(input_shape, num_classes):
    inp = Input(shape=input_shape)
    # Normalize only across channel axis (last dim)
    norm = Normalization(axis=-1)
    x = norm(inp)
    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)
    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, norm


if __name__ == "__main__":
    # 1) Load and split data
    X, y = load_flow_data()
    num_classes = len(np.unique(y))
    print(f"Loaded {len(X)} samples, {num_classes} classes.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # 2) Build model & adapt Normalization
    model, norm_layer = build_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    norm_layer.adapt(X_train)
    model.summary()

    # 3) Create datasets
    train_ds = build_dataset(X_train, y_train, augment=True)
    val_ds   = build_dataset(X_val,   y_val)
    test_ds  = build_dataset(X_test,  y_test)

    # 4) Train
    es = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[es]
    )

    # 5) Evaluate and report
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    ticks = np.arange(num_classes)
    plt.xticks(ticks, ticks+1)
    plt.yticks(ticks, ticks+1)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix'); plt.colorbar(); plt.tight_layout(); plt.show()

    # 6) Plot training curves
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy over Epochs'); plt.tight_layout(); plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss over Epochs'); plt.tight_layout(); plt.show()

    # 7) Save model
    model.save("material_model3.h5")
    print("Saved model to material_model3.h5")



