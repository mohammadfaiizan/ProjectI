"""
install_pytorch.py

Verify PyTorch installation and check CUDA compatibility.
Run this file first to ensure your environment is set up correctly.

Steps:
1. Check PyTorch installation
2. Verify PyTorch version
3. Check CUDA availability
4. Provide installation instructions if missing
"""

# Import necessary libraries (may fail if PyTorch not installed)
try:
    import torch
except ImportError:
    raise ImportError("PyTorch not found! See installation instructions below.")

def print_installation_instructions():
    """Display PyTorch installation commands for different setups"""
    print("\n" + "="*50)
    print("PYTORCH INSTALLATION INSTRUCTIONS")
    print("="*50)
    print("1. Visit https://pytorch.org/get-started/locally/")
    print("2. Select your environment (OS, package manager, CUDA version)")
    print("3. Run the recommended command. Examples:")
    print("\nFor CPU-only version:")
    print("pip install torch torchvision torchaudio")
    print("\nFor CUDA 11.8 (Windows/Linux):")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor Mac (M1/M2 Metal acceleration):")
    print("pip install torch torchvision torchaudio")

def main():
    """Main verification function"""
    # Basic PyTorch info
    print("="*50)
    print(f"PyTorch Version: {torch.__version__}")
    print("="*50)
    
    # CUDA availability check
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {'✅' if cuda_available else '❌'}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA Devices: {torch.cuda.device_count()}")
    else:
        print("\nNote: CUDA is not available. This means:")
        print("- You can still use PyTorch with CPU")
        print("- For GPU acceleration, ensure you have:")
        print("  a) NVIDIA GPU with CUDA support")
        print("  b) Correct NVIDIA drivers installed")
        print("  c) CUDA-compatible PyTorch version installed")

    # Performance optimization availability
    print("\n" + "="*50)
    print("Optimization Backends:")
    print(f"MPS (Apple Silicon) Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    print(f"OpenMP Available: {torch.backends.openmp.is_available()}")

if __name__ == "__main__":
    try:
        main()
    except NameError:
        print("\n" + "="*50)
        print("PYTORCH NOT INSTALLED!")
        print("="*50)
        print_installation_instructions()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print_installation_instructions()