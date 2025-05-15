#!/usr/bin/env python3
"""
frame_classify.py
=================

Image‑based GelSight material classification.

Usage (from the *data_root* directory):
    python frame_classify.py              # defaults: B0 backbone, 224×224, 32‑batch   b1 to b4 for larger slower models, have RN50 and RN101
    python frame_classify.py --help       # show all CLI flags
    python frame_classify.py --backbone B2 --img_size 256 --batch 16 --epochs 200
    python frame_classify.py --mixup 0.4      # stronger 




    cpu:   python frame_classify.py --backbone B0 --img_size 192 --batch 8 --mixup 0 --epochs 60
"""

# ---- imports -----------------------------------------------------------------
import os, glob, datetime, argparse
import numpy as np, tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Dense, Dropout,
                                     GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.applications import efficientnet_v2, resnet
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ---- CLI ---------------------------------------------------------------------
PAR = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
PAR.add_argument("--img_size",  type=int, default=224,   help="Square input size")
PAR.add_argument("--batch",     type=int, default=32,    help="Batch size")
PAR.add_argument("--epochs",    type=int, default=120,   help="Max epochs (with ES)")
PAR.add_argument("--backbone",  type=str, default="B0",
                 help="Backbone: B0/B1/B2/B3/B4 or RN50/RN101")
PAR.add_argument("--mixup",     type=float, default=0.2, help="MixUp α (0 disables)")
args = PAR.parse_args()

IMG_SIZE    = args.img_size
BATCH       = args.batch
EPOCHS      = args.epochs
BACKBONE    = args.backbone.upper()
MIXUP_ALPHA = args.mixup
SEED        = 42

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except ValueError:
    pass                               # CPU‑only TF

# ---- focal‑loss --------------------------------------------------------------
def focal_loss(γ=2.0, α=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        weight = α * K.pow(1 - y_pred, γ)
        return K.sum(weight * ce, axis=-1)
    return loss_fn

# ---- discover PNGs -----------------------------------------------------------
paths, labels = [], []
for mdir in sorted(glob.glob("material_*")):
    if not (os.path.isdir(mdir) and mdir.split("_")[-1].isdigit()): continue
    lab = int(mdir.split("_")[-1]) - 1         # zero‑based
    for png in glob.glob(os.path.join(mdir, "cycle_*", "frame.png")):
        paths.append(png); labels.append(lab)

paths, labels = np.asarray(paths), np.asarray(labels, np.int32)
n_classes      = labels.max() + 1
print(f"Found {len(paths)} images across {n_classes} classes.")

# ---- splits ------------------------------------------------------------------
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(
    X_tmp,  y_tmp,  test_size=0.5, stratify=y_tmp, random_state=SEED)

# ---- tf.data helpers ---------------------------------------------------------
def decode(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.cast(img, tf.float32)

def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.25)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_hue(img, 0.05)
    img = tf.image.random_crop(tf.image.pad_to_bounding_box(
          img, 4, 4, IMG_SIZE+8, IMG_SIZE+8), (IMG_SIZE, IMG_SIZE, 3))
    return img

def mixup(bx, by, α):
    lam = tf.random.gamma([], α, 1.0)
    idx = tf.random.shuffle(tf.range(tf.shape(bx)[0]))
    return lam*bx + (1-lam)*tf.gather(bx, idx), \
           lam*by + (1-lam)*tf.gather(by, idx)

def build_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    def _proc(p, y):
        img = decode(p)
        if training: img = augment(img)
        img = preprocess(img)
        return img, tf.one_hot(y, n_classes)
    ds = ds.map(_proc, tf.data.AUTOTUNE)
    ds = ds.batch(BATCH)
    if training and MIXUP_ALPHA > 0:
        ds = ds.map(lambda x,y: mixup(x,y,MIXUP_ALPHA), tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

# ---- backbone selection ------------------------------------------------------
def build_backbone(name):
    name = name.upper()
    if name.startswith("B"):                     # EfficientNetV2 family
        idx = {"B0":0,"B1":1,"B2":2,"B3":3,"B4":4}[name]
        model_fn = [efficientnet_v2.EfficientNetV2B0,
                    efficientnet_v2.EfficientNetV2B1,
                    efficientnet_v2.EfficientNetV2B2,
                    efficientnet_v2.EfficientNetV2B3,
                    efficientnet_v2.EfficientNetV2S][idx]
        bb = model_fn(include_top=False, weights="imagenet",
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
        preprocess = efficientnet_v2.preprocess_input
    elif name.startswith("RN"):                  # ResNet family
        if name=="RN50":
            bb = resnet.ResNet50(include_top=False, weights="imagenet",
                                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
        elif name=="RN101":
            bb = resnet.ResNet101(include_top=False, weights="imagenet",
                                  input_shape=(IMG_SIZE, IMG_SIZE, 3))
        else:
            raise ValueError("Unsupported backbone "+name)
        preprocess = resnet.preprocess_input
    else:
        raise ValueError("Unknown backbone "+name)
    return bb, preprocess

backbone, preprocess = build_backbone(BACKBONE)
backbone.trainable   = False

# ---- datasets ----------------------------------------------------------------
train_ds = build_ds(X_tr,  y_tr,  True)
val_ds   = build_ds(X_val, y_val, False)
test_ds  = build_ds(X_te,  y_te,  False)

# ---- class‑weights to balance ------------------------------------------------
cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels)
class_w = {i:w for i,w in enumerate(cw)}

# ---- model head --------------------------------------------------------------
inputs  = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = backbone(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = Dropout(0.4)(x)
outputs = Dense(n_classes, activation="softmax", dtype="float32")(x)
model   = Model(inputs, outputs)

model.compile(
    optimizer=AdamW(1e-3, weight_decay=1e-4),
    loss=focal_loss(),
    metrics=["accuracy"])

model.summary()

# ---- callbacks ---------------------------------------------------------------
tag   = f"{BACKBONE}_{IMG_SIZE}"
tlog  = f"logs/{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
cbs   = [EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
         ReduceLROnPlateau(patience=6, factor=0.3, min_lr=1e-6, monitor="val_loss"),
         ModelCheckpoint(f"best_{tag}.keras", save_best_only=True, monitor="val_loss"),
         TensorBoard(log_dir=tlog)]

# ---- stage‑1 training (frozen backbone) --------------------------------------
model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
          class_weight=class_w, callbacks=cbs)

# ---- fine‑tune upper half ----------------------------------------------------
for layer in backbone.layers[len(backbone.layers)//2:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=AdamW(1e-4, weight_decay=1e-4),
    loss=focal_loss(),
    metrics=["accuracy"])

model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds,
          class_weight=class_w, callbacks=cbs)

# ---- evaluation --------------------------------------------------------------
te_loss, te_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy: {te_acc*100:.2f}%")

# ---- save final --------------------------------------------------------------
model.save(f"material_img_model_{tag}.keras")
print(f"✔  Saved → material_img_model_{tag}.keras")
