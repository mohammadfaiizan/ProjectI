"""
Neural style transfer (content loss, style loss via Gram matrix).
"""
import tensorflow as tf

def get_feature_extractor(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model(vgg.input, outputs)

def gram_matrix(x):
    x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
    n = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], tf.float32)
    gram = tf.matmul(x, x, transpose_a=True) / n
    return gram

def content_loss(content, target):
    return tf.reduce_mean(tf.square(content - target))

def style_loss(style, target):
    g_style = gram_matrix(style)
    g_target = gram_matrix(target)
    return tf.reduce_mean(tf.square(g_style - g_target))

def main():
    layer_names = ['block1_conv1', 'block2_conv2', 'block3_conv3', 'block4_conv4']
    extractor = get_feature_extractor(layer_names)
    content_img = tf.random.normal((1, 224, 224, 3))
    style_img = tf.random.normal((1, 224, 224, 3))
    content_features = extractor(content_img)
    style_features = extractor(style_img)
    print(f"Number of feature layers: {len(content_features)}")
    for i, f in enumerate(content_features):
        print(f"Layer {i} shape: {f.shape}")
    c_loss = content_loss(content_features[-1], content_features[-1])
    s_loss = style_loss(style_features[0], style_features[0])
    print(f"Content loss: {c_loss.numpy():.4f}")
    print(f"Style loss: {s_loss.numpy():.4f}")
    g = gram_matrix(content_features[0])
    print(f"Gram matrix shape: {g.shape}")
    print("Style transfer components verified.")

if __name__ == "__main__":
    main()
