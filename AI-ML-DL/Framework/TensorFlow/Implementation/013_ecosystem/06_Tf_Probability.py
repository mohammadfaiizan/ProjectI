"""
TF Probability: tfd.Normal, tfd.MultivariateNormalDiag, tfp.layers, probabilistic models.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorFlow Probability")
    print("=" * 50)

    try:
        import tensorflow_probability as tfp
        tfd = tfp.distributions
        print(f"tensorflow_probability version: {tfp.__version__}")
    except ImportError:
        print("tensorflow_probability not installed. Install: pip install tensorflow-probability")
        return

    print("\ntfd.Normal:")
    normal = tfd.Normal(loc=0.0, scale=1.0)
    samples = normal.sample(5)
    log_prob = normal.log_prob(samples)
    print(f"  Samples: {samples.numpy()}")
    print(f"  Log prob: {log_prob.numpy()}")

    print("\ntfd.MultivariateNormalDiag:")
    mvn = tfd.MultivariateNormalDiag(loc=[0.0, 0.0], scale_diag=[1.0, 2.0])
    mvn_samples = mvn.sample(3)
    print(f"  Samples shape: {mvn_samples.shape}")

    print("\ntfp.layers - DistributionLambda:")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, input_shape=(4,)),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1e-5))
    ])
    x = tf.random.normal((8, 4))
    dist = model(x)
    print(f"  Output distribution: {type(dist).__name__}")
    print(f"  Sample shape: {dist.sample().shape}")

    print("\nProbabilistic regression (mean + std):")
    def negloglik(y_true, y_pred):
        return -y_pred.log_prob(y_true)
    inputs = tf.keras.Input(shape=(4,))
    hidden = tf.keras.layers.Dense(8, activation="relu")(inputs)
    params = tf.keras.layers.Dense(2)(hidden)
    dist_layer = tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1], scale=tf.nn.softplus(t[..., 1:]) + 1e-5)
    )(params)
    prob_model = tf.keras.Model(inputs, dist_layer)
    prob_model.compile(optimizer="adam", loss=negloglik)
    print("  Probabilistic model built with negative log-likelihood loss")

    print("\nTF Probability demo complete.")

if __name__ == "__main__":
    main()
