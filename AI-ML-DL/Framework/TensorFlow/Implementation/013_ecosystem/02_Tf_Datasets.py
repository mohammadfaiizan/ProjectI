"""
TF Datasets: tfds.load, tfds.builder, catalog, splits, as_supervised.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorFlow Datasets")
    print("=" * 50)

    try:
        import tensorflow_datasets as tfds
        print(f"tensorflow_datasets version: {tfds.__version__}")
    except ImportError:
        print("tensorflow_datasets not installed. Install: pip install tensorflow-datasets")
        return

    print("\nAvailable datasets (sample):")
    builder = tfds.builder("mnist")
    print(f"  Dataset: {builder.info.name}")
    print(f"  Description: {builder.info.description[:60]}...")
    print(f"  Splits: {list(builder.info.splits.keys())}")

    print("\ntfds.load with as_supervised:")
    ds, info = tfds.load("mnist", split="train", as_supervised=True, with_info=True)
    print(f"  Split: train, samples: {info.splits['train'].num_examples}")
    for img, label in ds.take(1):
        print(f"  Sample: image {img.shape}, label {label.numpy()}")

    print("\nBuilder info:")
    print(f"  Features: {builder.info.features}")
    print(f"  Homepage: {builder.info.homepage}")

    print("\nIterating splits:")
    for split_name, split_info in builder.info.splits.items():
        print(f"  {split_name}: {split_info.num_examples} examples")

    print("\nTF Datasets demo complete.")

if __name__ == "__main__":
    main()
