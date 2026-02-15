"""
Transfer learning with tf.keras.applications (ResNet50, MobileNet).
"""
import tensorflow as tf

def build_resnet50_classifier(num_classes=10, input_shape=(224, 224, 3)):
    base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_mobilenet_classifier(num_classes=10, input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    resnet = build_resnet50_classifier()
    resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x = tf.random.normal((2, 224, 224, 3))
    y_resnet = resnet(x)
    print(f"ResNet50 output shape: {y_resnet.shape}")
    print(f"ResNet50 params: {resnet.count_params():,}")
    mobilenet = build_mobilenet_classifier()
    y_mobilenet = mobilenet(x)
    print(f"MobileNet output shape: {y_mobilenet.shape}")
    print(f"MobileNet params: {mobilenet.count_params():,}")
    base = mobilenet.layers[0]
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    print(f"Fine-tuning last 20 layers of MobileNet")
    print("Transfer learning models built successfully.")

if __name__ == "__main__":
    main()
