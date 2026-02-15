#!/usr/bin/env python3
"""PyTorch Installation and Environment Setup Verification"""

import sys
import os

def check_python_version():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 8:
        print("✅ Python version compatible with PyTorch")
    else:
        print("❌ Python 3.8+ required for latest PyTorch")

def install_pytorch():
    print("\n=== PyTorch Installation Commands ===")
    print("# CPU-only installation:")
    print("pip install torch torchvision torchaudio")
    print("\n# CUDA 11.8 installation:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n# CUDA 12.1 installation:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

def check_pytorch_installation():
    try:
        import torch
        print(f"\n✅ PyTorch installed: {torch.__version__}")
        return True
    except ImportError:
        print("\n❌ PyTorch not installed")
        return False

def check_cuda_availability():
    try:
        import torch
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available - using CPU")
    except Exception as e:
        print(f"Error checking CUDA: {e}")

def check_additional_packages():
    packages = {
        'torchvision': 'Computer vision utilities',
        'torchaudio': 'Audio processing utilities',
        'numpy': 'Numerical computing',
        'matplotlib': 'Plotting and visualization',
        'PIL': 'Image processing'
    }
    
    print("\n=== Additional Package Status ===")
    for package, description in packages.items():
        try:
            if package == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package}: {version} - {description}")
        except ImportError:
            print(f"❌ {package}: Not installed - {description}")

def test_basic_operations():
    try:
        import torch
        print("\n=== Basic PyTorch Test ===")
        
        # Create tensors
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        print(f"Tensor x shape: {x.shape}")
        print(f"Tensor y shape: {y.shape}")
        
        # Basic operations
        z = x + y
        print(f"Addition result shape: {z.shape}")
        
        # GPU test if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = x_gpu + y_gpu
            print(f"GPU operation successful: {z_gpu.device}")
        
        print("✅ Basic operations test passed")
        
    except Exception as e:
        print(f"❌ Basic operations test failed: {e}")

def check_environment_variables():
    print("\n=== Environment Variables ===")
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: Not set")

def performance_check():
    try:
        import torch
        import time
        
        print("\n=== Performance Check ===")
        
        # CPU performance
        start_time = time.time()
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.mm(x, y)
        cpu_time = time.time() - start_time
        print(f"CPU matrix multiplication (1000x1000): {cpu_time:.4f} seconds")
        
        # GPU performance if available
        if torch.cuda.is_available():
            x_gpu = torch.randn(1000, 1000).cuda()
            y_gpu = torch.randn(1000, 1000).cuda()
            torch.cuda.synchronize()
            start_time = time.time()
            z_gpu = torch.mm(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            print(f"GPU matrix multiplication (1000x1000): {gpu_time:.4f} seconds")
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"GPU speedup: {speedup:.2f}x")
        
    except Exception as e:
        print(f"Performance check failed: {e}")

def main():
    print("PyTorch Environment Setup and Verification")
    print("=" * 50)
    
    check_python_version()
    install_pytorch()
    
    if check_pytorch_installation():
        check_cuda_availability()
        check_additional_packages()
        test_basic_operations()
        check_environment_variables()
        performance_check()
    else:
        print("\nPlease install PyTorch first using the commands above")

if __name__ == "__main__":
    main() 