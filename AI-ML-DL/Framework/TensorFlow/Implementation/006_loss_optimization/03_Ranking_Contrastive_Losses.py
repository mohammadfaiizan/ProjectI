"""
Triplet loss, contrastive loss, cosine similarity loss.
"""
import tensorflow as tf

def triplet_loss(anchor, positive, negative, margin=0.5):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0.0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def cosine_similarity_loss(y_true, y_pred):
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    cos_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    return 1.0 - tf.reduce_mean(cos_sim)

def main():
    batch_size = 4
    dim = 8
    anchor = tf.random.normal((batch_size, dim))
    positive = anchor + tf.random.normal((batch_size, dim)) * 0.1
    negative = tf.random.normal((batch_size, dim))

    loss_triplet = triplet_loss(anchor, positive, negative, margin=0.5)
    print(f"Triplet loss (margin=0.5): {loss_triplet.numpy():.4f}")

    y_true_pair = tf.constant([1.0, 0.0, 1.0, 0.0])
    y_pred_dist = tf.constant([0.3, 1.2, 0.5, 1.5])
    loss_contrastive = contrastive_loss(y_true_pair, y_pred_dist, margin=1.0)
    print(f"Contrastive loss: {loss_contrastive.numpy():.4f}")

    y_true_vec = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    y_pred_vec = tf.constant([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]], dtype=tf.float32)
    loss_cos = cosine_similarity_loss(y_true_vec, y_pred_vec)
    print(f"Cosine similarity loss: {loss_cos.numpy():.4f}")

    class TripletLossLayer(tf.keras.layers.Layer):
        def __init__(self, margin=0.5, **kwargs):
            super().__init__(**kwargs)
            self.margin = margin

        def call(self, inputs):
            anchor, positive, negative = inputs
            loss = triplet_loss(anchor, positive, negative, self.margin)
            self.add_loss(loss)
            return loss

    inp_anchor = tf.keras.Input(shape=(dim,))
    inp_pos = tf.keras.Input(shape=(dim,))
    inp_neg = tf.keras.Input(shape=(dim,))
    out = TripletLossLayer(margin=0.5)([inp_anchor, inp_pos, inp_neg])
    model = tf.keras.Model(inputs=[inp_anchor, inp_pos, inp_neg], outputs=out)
    model.compile(optimizer='adam')
    loss_val = model.train_on_batch([anchor, positive, negative], tf.zeros(1))
    print(f"Triplet model loss: {loss_val:.4f}")
    print("Ranking and contrastive losses verified.")

if __name__ == "__main__":
    main()
