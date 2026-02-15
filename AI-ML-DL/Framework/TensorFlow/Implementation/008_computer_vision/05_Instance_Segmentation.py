"""
Mask-based instance segmentation approach.
"""
import tensorflow as tf

def build_instance_segmentation_head(backbone_output, num_classes=21, num_instances=5):
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(backbone_output)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    cls_logits = tf.keras.layers.Conv2D(num_classes, 1, activation='linear')(x)
    mask_logits = tf.keras.layers.Conv2D(num_instances, 1, activation='sigmoid')(x)
    return cls_logits, mask_logits

def build_instance_seg_model(input_shape=(128, 128, 3), num_classes=21, num_instances=5):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(inp)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
    backbone = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    cls_out, mask_out = build_instance_segmentation_head(backbone, num_classes, num_instances)
    model = tf.keras.Model(inp, [cls_out, mask_out])
    return model

def mask_iou(mask1, mask2):
    intersection = tf.reduce_sum(mask1 * mask2)
    union = tf.reduce_sum(mask1) + tf.reduce_sum(mask2) - intersection
    return intersection / (union + 1e-6)

def main():
    model = build_instance_seg_model()
    x = tf.random.normal((2, 128, 128, 3))
    cls_pred, mask_pred = model(x)
    print(f"Class logits shape: {cls_pred.shape}")
    print(f"Mask predictions shape: {mask_pred.shape}")
    m1 = tf.random.uniform((2, 32, 32, 1), 0, 1)
    m2 = tf.random.uniform((2, 32, 32, 1), 0, 1)
    iou = mask_iou(m1, m2)
    print(f"Mask IoU: {iou.numpy()}")
    print("Instance segmentation model built.")

if __name__ == "__main__":
    main()
