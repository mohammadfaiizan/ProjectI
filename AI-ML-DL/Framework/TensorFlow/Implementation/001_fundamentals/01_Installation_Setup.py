"""
TensorFlow Installation and Environment Verification
"""
import sys

def main():
    print("=" * 50)
    print("TensorFlow Installation Check")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"TensorFlow not installed: {e}")
        sys.exit(1)
    
    print("\n--- GPU Setup ---")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
        print(f"Total GPUs available: {len(gpus)}")
    else:
        print("No GPU devices found. Using CPU.")
    
    print("\n--- Environment Verification ---")
    print(f"Python version: {sys.version}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available (test): {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    print("\n--- Quick Tensor Test ---")
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    print(f"Test tensor:\n{x}")
    print(f"Device placement: {x.device}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
