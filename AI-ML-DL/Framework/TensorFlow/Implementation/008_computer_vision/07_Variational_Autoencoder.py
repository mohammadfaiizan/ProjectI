"""
VAE (encoder, decoder, reparameterization, KL loss).
"""
import tensorflow as tf

def build_vae_encoder(input_shape=(28, 28, 1), latent_dim=20):
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inp)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)
    return tf.keras.Model(inp, [z_mean, z_log_var])

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae_decoder(latent_dim=20, output_shape=(28, 28, 1)):
    inp = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(inp)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    out = tf.keras.layers.Conv2D(output_shape[-1], 3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inp, out)

def vae_loss(x, x_recon, z_mean, z_log_var):
    recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_recon)) * 28 * 28
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return recon_loss + kl_loss

def main():
    latent_dim = 20
    encoder = build_vae_encoder(latent_dim=latent_dim)
    decoder = build_vae_decoder(latent_dim=latent_dim)
    inp = tf.keras.layers.Input(shape=(28, 28, 1))
    z_mean, z_log_var = encoder(inp)
    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    out = decoder(z)
    vae = tf.keras.Model(inp, out)
    x = tf.random.uniform((4, 28, 28, 1), 0, 1)
    x_recon = vae(x)
    print(f"Reconstruction shape: {x_recon.shape}")
    loss = vae_loss(x, x_recon, z_mean, z_log_var)
    print(f"VAE loss: {loss.numpy():.4f}")
    print("VAE built successfully.")

if __name__ == "__main__":
    main()
