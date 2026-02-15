# Knowledge Distillation and Self-Supervised Learning

## Table of Contents

1. [Knowledge Distillation](#1-knowledge-distillation)
2. [Self-Supervised Learning](#2-self-supervised-learning)
3. [Contrastive Learning](#3-contrastive-learning)

---

## 1. Knowledge Distillation

**Knowledge distillation** transfers knowledge from a large **teacher** model to a smaller **student** model. The student learns from both hard labels and soft probability distributions produced by the teacher.

### Soft Targets and Temperature

The teacher produces a **softmax** over classes. A higher **temperature** T produces softer (smoother) distributions that reveal dark knowledge (similarities between classes):

```python
def softmax_with_temperature(logits, temperature):
    return tf.nn.softmax(logits / temperature)
```

At T=1, softmax is standard. At T>1, probabilities become more uniform; at T<1, they become sharper.

### Distillation Loss

Combine **KL divergence** between teacher and student soft outputs with **cross-entropy** on hard labels:

```python
def distillation_loss(y_true, logits_student, logits_teacher, temperature=4.0, alpha=0.7):
    soft_teacher = tf.nn.softmax(logits_teacher / temperature)
    soft_student = tf.nn.softmax(logits_student / temperature)
    kl = tf.reduce_mean(
        tf.reduce_sum(soft_teacher * (tf.math.log(soft_teacher + 1e-8) - tf.math.log(soft_student + 1e-8)), axis=1)
    )
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, logits_student)
    return alpha * (temperature ** 2) * kl + (1 - alpha) * tf.reduce_mean(hard_loss)
```

The `temperature ** 2` factor corrects for the gradient scale when using soft targets.

### Training Loop

```python
for x, y in ds:
    with tf.GradientTape() as tape:
        logits_teacher = teacher(x, training=False)
        logits_student = student(x, training=True)
        loss = distillation_loss(y, logits_student, logits_teacher, T=4.0)
    grads = tape.gradient(loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
```

### Use Cases

- **Model compression**: Large model to small model.
- **Ensemble distillation**: Multiple teachers to one student.
- **Cross-modal**: Image teacher to text student.

---

## 2. Self-Supervised Learning

**Self-supervised learning** learns representations from unlabeled data by defining **pretext tasks** that generate pseudo-labels from the data itself.

### Pretext Tasks

| Task | Input | Label | Example |
|------|-------|-------|---------|
| **Rotation prediction** | Rotated image | Rotation angle (0, 90, 180, 270) | RotNet |
| **Jigsaw puzzle** | Shuffled patches | Permutation index | Jigsaw |
| **Colorization** | Grayscale image | Original color | Colorization |
| **Masked prediction** | Masked sequence | Masked tokens | BERT, MAE |

### Rotation Prediction

```python
def rotation_pretext(images):
    rotations = [0, 1, 2, 3]  # 0, 90, 180, 270
    batch_x, batch_y = [], []
    for k in rotations:
        rot_img = tf.image.rot90(images, k=k)
        batch_x.append(rot_img)
        batch_y.extend([k] * tf.shape(images)[0])
    return tf.concat(batch_x, axis=0), tf.constant(batch_y)
```

The encoder learns features useful for predicting rotation, which encourages understanding of object structure and orientation.

### Training

```python
encoder = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 rotation classes
])
encoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
x_aug, y_aug = rotation_pretext(images)
encoder.fit(x_aug, y_aug, epochs=10)
```

The convolutional layers can be used as a frozen feature extractor for downstream tasks.

---

## 3. Contrastive Learning

**Contrastive learning** learns representations by pulling together positive pairs (same semantic content) and pushing apart negative pairs (different content) in embedding space.

### SimCLR Framework

1. **Augmentation**: Create two augmented views of each image.
2. **Encoder**: Map to representation.
3. **Projection head**: Map to lower-dimensional space for contrastive loss.
4. **NT-Xent loss**: InfoNCE / normalized temperature-scaled cross-entropy.

### NT-Xent Loss

For a batch of N samples, create 2N augmented views. For each anchor, its augmented pair is the positive; the other 2N-2 are negatives:

```python
def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = tf.concat([z_i, z_j], axis=0)
    z = tf.math.l2_normalize(z, axis=1)
    sim = tf.matmul(z, z, transpose_b=True) / temperature
    n = tf.shape(z_i)[0]
    mask = tf.eye(2 * n)
    sim = sim - 1e9 * mask
    labels = tf.concat([tf.range(n, 2*n), tf.range(n)], axis=0)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, sim, from_logits=True)
    return tf.reduce_mean(loss)
```

### Projection Head

A small MLP maps encoder outputs to the contrastive space:

```python
projection = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64)
])
z = projection(encoder(x))
```

### Augmentation Strategy

Strong augmentations (crop, color jitter, blur) are critical. For images: random crop and resize, color jitter, random horizontal flip, sometimes Gaussian blur.

### Downstream Evaluation

Train a linear classifier on frozen encoder features. High linear probe accuracy indicates good representations.
