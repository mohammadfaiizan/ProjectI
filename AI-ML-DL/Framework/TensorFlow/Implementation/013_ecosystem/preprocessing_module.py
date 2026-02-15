"""
Preprocessing module for TFX Transform (used by 09_Tfx_Transform.py).
"""
import tensorflow as tf
import tensorflow_transform as tft


def preprocessing_fn(inputs):
    x = inputs["x"]
    x_normalized = tft.scale_to_z_score(x)
    x_bucketized = tft.bucketize(x_normalized, num_buckets=10)
    return {"x_normalized": x_normalized, "x_bucketized": x_bucketized}
