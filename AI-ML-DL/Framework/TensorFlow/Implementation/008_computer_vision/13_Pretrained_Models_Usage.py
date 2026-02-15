"""
tf.keras.applications catalog, feature extraction, fine-tuning.
"""
import tensorflow as tf

AVAILABLE_MODELS = [
    'ResNet50', 'ResNet101', 'ResNet152',
    'VGG16', 'VGG19',
    'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'MobileNetV3Large',
    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB7',
    'DenseNet121', 'DenseNet169', 'DenseNet201',
    'InceptionV3', 'InceptionResNetV2',
    'NASNetMobile', 'NASNetLarge',
    'Xception'
]

def get_pretrained_base(model_name='ResNet50', input_shape=(224, 224, 3), include_top=False):
    builders = {
        'ResNet50': tf.keras.applications.ResNet50,
        'ResNet101': tf.keras.applications.ResNet101,
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
        'VGG16': tf.keras.applications.VGG16,
        'DenseNet121': tf.keras.applications.DenseNet121,
    }
    if model_name not in builders:
        model_name = 'ResNet50'
    return builders[model_name](weights='imagenet', include_top=include_top, input_shape=input_shape)

def feature_extraction_model(base_name='ResNet50', num_classes=10):
    base = get_pretrained_base(base_name)
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def fine_tuning_model(base_name='MobileNetV2', num_classes=10, trainable_layers=20):
    base = get_pretrained_base(base_name)
    base.trainable = True
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    print("Available models:", len(AVAILABLE_MODELS))
    feat_model = feature_extraction_model('ResNet50')
    x = tf.random.normal((2, 224, 224, 3))
    y = feat_model(x)
    print(f"Feature extraction output: {y.shape}")
    ft_model = fine_tuning_model('MobileNetV2', trainable_layers=20)
    trainable = sum(1 for l in ft_model.trainable_weights)
    print(f"Fine-tuning trainable params count: {trainable}")
    base = get_pretrained_base('EfficientNetB0')
    print(f"EfficientNetB0 output shape: {base(x).shape}")
    print("Pretrained models usage verified.")

if __name__ == "__main__":
    main()
