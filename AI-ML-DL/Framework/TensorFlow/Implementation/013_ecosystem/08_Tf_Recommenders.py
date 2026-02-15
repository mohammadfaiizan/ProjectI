"""
TF Recommenders: tfrs.Model, retrieval, ranking tasks.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorFlow Recommenders")
    print("=" * 50)

    try:
        import tensorflow_recommenders as tfrs
        print(f"tensorflow_recommenders version: {tfrs.__version__}")
    except ImportError:
        print("tensorflow_recommenders not installed. Install: pip install tensorflow-recommenders")
        return

    print("\nRetrieval model concept:")
    user_ids = ["u1", "u2", "u3"]
    item_ids = ["i1", "i2", "i3", "i4"]
    user_embedding_dim = 32
    item_embedding_dim = 32

    user_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(user_ids) + 1, user_embedding_dim)
    ])
    item_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(vocabulary=item_ids, mask_token=None),
        tf.keras.layers.Embedding(len(item_ids) + 1, item_embedding_dim)
    ])

    class RetrievalModel(tfrs.Model):
        def __init__(self):
            super().__init__()
            self.user_model = user_model
            self.item_model = item_model
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=tf.data.Dataset.from_tensor_slices(item_ids).batch(4).map(item_model)
                )
            )

        def compute_loss(self, features, training=False):
            user_embeddings = self.user_model(features["user_id"])
            item_embeddings = self.item_model(features["item_id"])
            return self.task(user_embeddings, item_embeddings)

    retrieval_model = RetrievalModel()
    retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
    print("  Retrieval model with FactorizedTopK metrics")

    print("\nRanking model concept:")
    class RankingModel(tfrs.Model):
        def __init__(self):
            super().__init__()
            self.rating_model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(1)
            ])
            self.task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

        def call(self, features):
            return self.rating_model(tf.concat([
                tf.reshape(features["user_embedding"], (-1, 32)),
                tf.reshape(features["item_embedding"], (-1, 32))
            ], axis=1))

        def compute_loss(self, features, training=False):
            labels = features.pop("rating")
            ratings = self(features)
            return self.task(labels, ratings)

    ranking_model = RankingModel()
    ranking_model.compile(optimizer="adam")
    print("  Ranking model with MSE loss")

    print("\nTF Recommenders demo complete.")

if __name__ == "__main__":
    main()
