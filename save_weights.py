import tensorflow as tf
import numpy as np

old_model = tf.keras.models.load_model("alzheimer_model.h5", compile=False)
weights = old_model.get_weights()

# Save as object array to handle inhomogeneous shapes
np.save("model_weights.npy", np.array(weights, dtype=object), allow_pickle=True)
print("Done!")