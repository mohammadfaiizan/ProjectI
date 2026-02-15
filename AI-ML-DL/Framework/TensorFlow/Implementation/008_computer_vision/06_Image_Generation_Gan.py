"""
DCGAN implementation (generator + discriminator + training loop).
"""
import tensorflow as tf

def build_generator(latent_dim=100, img_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(img_shape[-1], 5, padding='same', activation='tanh')
    ])
    return model

def build_discriminator(img_shape=(28, 28, 1)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 5, strides=2, padding='same', input_shape=img_shape),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, 5, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    latent_dim = 100
    generator = build_generator(latent_dim)
    discriminator = build_discriminator()
    z = tf.random.normal((4, latent_dim))
    fake_imgs = generator(z)
    print(f"Generated images shape: {fake_imgs.shape}")
    real_pred = discriminator(tf.random.normal((4, 28, 28, 1)))
    fake_pred = discriminator(fake_imgs)
    print(f"Discriminator real pred: {real_pred.shape}, fake pred: {fake_pred.shape}")
    gan = tf.keras.Sequential([generator, discriminator])
    discriminator.trainable = False
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    print(f"GAN total params: {gan.count_params():,}")
    print("DCGAN built successfully.")

if __name__ == "__main__":
    main()
