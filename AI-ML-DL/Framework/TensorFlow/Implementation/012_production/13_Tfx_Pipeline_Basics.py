"""
TFX components overview (ExampleGen, Transform, Trainer, Evaluator, Pusher).
"""
import os

def main():
    print("=" * 50)
    print("TFX Pipeline Basics")
    print("=" * 50)

    components = [
        ("ExampleGen", "Ingests data from CSV, TFRecord, BigQuery"),
        ("Transform", "Preprocessing, feature engineering, schema"),
        ("Trainer", "Trains model with TF/Keras"),
        ("Evaluator", "Validates model metrics against baseline"),
        ("Pusher", "Deploys model to TF Serving or other targets")
    ]
    print("TFX Pipeline Components:")
    for name, desc in components:
        print(f"  {name}: {desc}")

    print("\nPipeline DAG: ExampleGen -> Transform -> Trainer -> Evaluator -> Pusher")
    print("  (Transform output feeds Trainer; Evaluator gates Pusher)")

    try:
        import tfx
        print(f"\nTFX version: {tfx.__version__}")
        from tfx.components import ExampleGen, Transform, Trainer, Evaluator, Pusher
        print("TFX components imported successfully")
    except ImportError:
        print("\nTFX not installed. Install: pip install tfx")

    pipeline_dir = os.path.join(os.path.dirname(__file__), "tfx_pipeline")
    os.makedirs(pipeline_dir, exist_ok=True)
    print(f"\nPipeline directory: {pipeline_dir}")

    print("\nTFX pipeline demo complete.")

if __name__ == "__main__":
    main()
