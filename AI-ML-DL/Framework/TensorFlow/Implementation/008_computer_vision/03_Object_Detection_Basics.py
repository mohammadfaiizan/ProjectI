"""
Basic object detection concepts and implementation.
"""
import tensorflow as tf

def build_detection_backbone(input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    return base

def build_simple_detector(num_classes=20, num_anchors=9):
    inp = tf.keras.layers.Input(shape=(224, 224, 3))
    backbone = build_detection_backbone()(inp)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(backbone)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    cls_out = tf.keras.layers.Conv2D(num_classes * num_anchors, 3, padding='same', activation='sigmoid')(x)
    box_out = tf.keras.layers.Conv2D(4 * num_anchors, 3, padding='same', activation='linear')(x)
    model = tf.keras.Model(inp, [cls_out, box_out])
    return model

def compute_iou(box1, box2):
    x1 = tf.maximum(box1[..., 0], box2[..., 0])
    y1 = tf.maximum(box1[..., 1], box2[..., 1])
    x2 = tf.minimum(box1[..., 2], box2[..., 2])
    y2 = tf.minimum(box1[..., 3], box2[..., 3])
    inter = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

def main():
    model = build_simple_detector()
    x = tf.random.normal((2, 224, 224, 3))
    cls_pred, box_pred = model(x)
    print(f"Classification output shape: {cls_pred.shape}")
    print(f"Bounding box output shape: {box_pred.shape}")
    box1 = tf.constant([[0.0, 0.0, 1.0, 1.0]])
    box2 = tf.constant([[0.5, 0.5, 1.5, 1.5]])
    iou = compute_iou(box1, box2)
    print(f"IoU example: {iou.numpy()}")
    print("Object detection basics verified.")

if __name__ == "__main__":
    main()
