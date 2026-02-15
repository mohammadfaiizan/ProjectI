"""
TFX Transform: preprocessing_fn, schema, Transform component.
"""
import os
import tensorflow as tf

def main():
    print("=" * 50)
    print("TFX Transform")
    print("=" * 50)

    try:
        from tfx.components import Transform
        import tensorflow_transform as tft
        print("TFX Transform imported successfully")
    except ImportError:
        print("tfx not installed. Install: pip install tfx")
        return

    def preprocessing_fn(inputs):
        x = inputs["x"]
        x_normalized = tft.scale_to_z_score(x)
        x_bucketized = tft.bucketize(x_normalized, num_buckets=10)
        return {"x_normalized": x_normalized, "x_bucketized": x_bucketized}

    print("\npreprocessing_fn:")
    print("  - scale_to_z_score: normalize to zero mean, unit variance")
    print("  - bucketize: discretize continuous values")

    print("\nSchema for Transform:")
    raw_schema = """
    feature {
      name: "x"
      type: FLOAT
    }
    """
    print(raw_schema)

    print("\nTransform component setup:")
    module_path = os.path.join(os.path.dirname(__file__), "preprocessing_module.py")
    print(f"  module_file: {module_path}")
    print("  Transform requires: examples (ExampleGen output), schema, module_file (preprocessing_fn)")

    print("\ntft analyzers (computed over full dataset):")
    print("  tft.scale_to_z_score, tft.bucketize, tft.vocabulary")
    print("  tft.compute_and_apply_vocabulary for categorical features")

    print("\nTransform outputs:")
    print("  transform_graph: SavedModel for preprocessing at serving")
    print("  transformed_examples: TFRecord with transformed features")

    print("\nTFX Transform demo complete.")

if __name__ == "__main__":
    main()
