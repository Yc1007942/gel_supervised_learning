import sys, json, os
import numpy as np
import tensorflow as tf

def load_flow(path):
    Ox, Oy, Cx, Cy, _ = json.load(open(path))
    dx = np.array(Cx) - np.array(Ox)
    dy = np.array(Cy) - np.array(Oy)
    dx = dx.reshape((6,8))
    dy = dy.reshape((6,8))
    return np.stack([dx, dy], axis=-1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_material.py path/to/flow.json")
        sys.exit(1)

    flow_path = sys.argv[1]
    if not os.path.isfile(flow_path):
        print("File not found:", flow_path)
        sys.exit(1)

    # 1) Load model
    model = tf.keras.models.load_model("material_model3.h5")

    # 2) Load & preprocess one sample
    x = load_flow(flow_path)
    x = x[np.newaxis, ...]     # shape (1,6,8,2)

    # 3) Predict
    probs = model.predict(x)[0]
    cls  = np.argmax(probs)
    conf = probs[cls]

    # 4) Report (add 1 if your materials are 1-indexed)
    print(f"Predicted material: {cls+1}  (confidence {conf*100:.1f}%)")
