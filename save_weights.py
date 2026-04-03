import tensorflow as tf
import numpy as np
# Load old model
old_model = tf.keras.models.load_model("alzheimer_model.h5", compile=False)
# Extract weights
weights = old_model.get_weights()
# Save as numpy
np.save("model_weights.npy", weights, allow_pickle=True)
print("Done! Push model_weights.npy to GitHub") 
