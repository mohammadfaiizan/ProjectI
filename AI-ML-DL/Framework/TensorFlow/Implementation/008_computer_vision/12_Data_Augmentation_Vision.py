"""
Advanced augmentation (Mixup, CutMix, RandAugment concepts).
"""
import tensorflow as tf
import numpy as np

def mixup(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0, alpha)
    lam = tf.maximum(lam, 1 - lam)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels

def cutmix_simple(images, labels, alpha=1.0):
    batch_size, H, W = images.shape[0], images.shape[1], images.shape[2]
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    cut_h = max(1, int(H * np.sqrt(1 - lam)))
    cut_w = max(1, int(W * np.sqrt(1 - lam)))
    cy = np.random.randint(0, max(1, H - cut_h + 1))
    cx = np.random.randint(0, max(1, W - cut_w + 1))
    indices = np.random.permutation(batch_size)
    mask = np.ones((batch_size, H, W, images.shape[-1]))
    mask[:, cy:cy+cut_h, cx:cx+cut_w, :] = 0
    mixed = images * mask + images[indices] * (1 - mask)
    mixed_labels = labels * lam + labels[indices] * (1 - lam)
    return mixed, mixed_labels

def rand_augment_layers():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 1))
    ])

def main():
    images = tf.random.normal((4, 32, 32, 3))
    labels = tf.one_hot(tf.constant([0, 1, 2, 3]), 10)
    mixed_imgs, mixed_lbls = mixup(images, labels)
    print(f"Mixup images: {mixed_imgs.shape}, labels: {mixed_lbls.shape}")
    imgs_np = images.numpy()
    lbls_np = labels.numpy()
    cut_imgs, cut_lbls = cutmix_simple(imgs_np, lbls_np)
    print(f"CutMix images: {cut_imgs.shape}")
    aug = rand_augment_layers()
    aug_imgs = aug(images)
    print(f"RandAugment output: {aug_imgs.shape}")
    print("Data augmentation verified.")

if __name__ == "__main__":
    main()
