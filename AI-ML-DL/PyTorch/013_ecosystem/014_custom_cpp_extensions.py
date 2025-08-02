import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import subprocess
import tempfile

# Custom C++/CUDA Extensions for PyTorch
# Note: This demonstrates C++/CUDA extension concepts and provides utilities

class CPPExtensionBuilder:
    """Builder for C++ PyTorch extensions"""
    
    def __init__(self, extension_name: str = "custom_extension"):
        self.extension_name = extension_name
        self.source_files = []
        self.include_dirs = []
        self.library_dirs = []
        self.libraries = []
        self.extra_compile_args = []
        self.extra_link_args = []
        
    def add_source_file(self, filepath: str):
        """Add C++ source file"""
        self.source_files.append(filepath)
        return self
    
    def add_include_dir(self, directory: str):
        """Add include directory"""
        self.include_dirs.append(directory)
        return self
    
    def add_library(self, library: str, library_dir: Optional[str] = None):
        """Add library to link"""
        self.libraries.append(library)
        if library_dir:
            self.library_dirs.append(library_dir)
        return self
    
    def add_compile_args(self, args: List[str]):
        """Add extra compile arguments"""
        self.extra_compile_args.extend(args)
        return self
    
    def generate_setup_py(self, output_dir: str) -> str:
        """Generate setup.py for the extension"""
        
        setup_py_content = f'''
import os
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
ext_modules = [
    Pybind11Extension(
        "{self.extension_name}",
        {self.source_files},
        include_dirs={self.include_dirs + ['$(python -c "import torch; print(torch.utils.cpp_extension.include_paths())")']},
        libraries={self.libraries},
        library_dirs={self.library_dirs},
        extra_compile_args={self.extra_compile_args},
        extra_link_args={self.extra_link_args},
        cxx_std=14,
    ),
]

setup(
    name="{self.extension_name}",
    version=__version__,
    author="PyTorch Extension Builder",
    description="Custom PyTorch C++ extension",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={{"build_ext": build_ext}},
    zip_safe=False,
    python_requires=">=3.6",
)
'''
        
        setup_path = Path(output_dir) / "setup.py"
        with open(setup_path, 'w') as f:
            f.write(setup_py_content)
        
        return str(setup_path)
    
    def generate_cpp_templates(self, output_dir: str) -> Dict[str, str]:
        """Generate C++ template files"""
        
        output_path = Path(output_dir)
        generated_files = {}
        
        # Main C++ file
        cpp_content = f'''
#include <torch/extension.h>
#include <vector>
#include <iostream>

// Forward declarations
torch::Tensor custom_add_forward(torch::Tensor input1, torch::Tensor input2);
std::vector<torch::Tensor> custom_add_backward(torch::Tensor grad_output);

// Custom addition operation
torch::Tensor custom_add_forward(torch::Tensor input1, torch::Tensor input2) {{
    // Check tensor properties
    TORCH_CHECK(input1.dtype() == input2.dtype(), "Input tensors must have the same dtype");
    TORCH_CHECK(input1.device() == input2.device(), "Input tensors must be on the same device");
    
    // Perform addition
    return input1 + input2;
}}

// Backward pass for custom addition
std::vector<torch::Tensor> custom_add_backward(torch::Tensor grad_output) {{
    // Gradient for addition is just the incoming gradient for both inputs
    return {{grad_output, grad_output}};
}}

// Custom matrix multiplication with timing
torch::Tensor custom_matmul(torch::Tensor input1, torch::Tensor input2) {{
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result = torch::matmul(input1, input2);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Custom matmul took: " << duration.count() << " microseconds" << std::endl;
    
    return result;
}}

// Element-wise operations
torch::Tensor custom_sigmoid(torch::Tensor input) {{
    return 1.0 / (1.0 + torch::exp(-input));
}}

torch::Tensor custom_relu(torch::Tensor input) {{
    return torch::clamp_min(input, 0);
}}

// Tensor utilities
torch::Tensor tensor_info(torch::Tensor input) {{
    std::cout << "Tensor info:" << std::endl;
    std::cout << "  Shape: " << input.sizes() << std::endl;
    std::cout << "  Dtype: " << input.dtype() << std::endl;
    std::cout << "  Device: " << input.device() << std::endl;
    std::cout << "  Requires grad: " << input.requires_grad() << std::endl;
    
    return input;
}}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("custom_add_forward", &custom_add_forward, "Custom addition forward pass");
    m.def("custom_add_backward", &custom_add_backward, "Custom addition backward pass");
    m.def("custom_matmul", &custom_matmul, "Custom matrix multiplication with timing");
    m.def("custom_sigmoid", &custom_sigmoid, "Custom sigmoid activation");
    m.def("custom_relu", &custom_relu, "Custom ReLU activation");
    m.def("tensor_info", &tensor_info, "Print tensor information");
}}
'''
        
        cpp_file = output_path / f"{self.extension_name}.cpp"
        with open(cpp_file, 'w') as f:
            f.write(cpp_content)
        generated_files["cpp_file"] = str(cpp_file)
        
        return generated_files

class CUDAExtensionBuilder(CPPExtensionBuilder):
    """Builder for CUDA PyTorch extensions"""
    
    def __init__(self, extension_name: str = "cuda_extension"):
        super().__init__(extension_name)
        self.cuda_files = []
        
    def add_cuda_file(self, filepath: str):
        """Add CUDA source file"""
        self.cuda_files.append(filepath)
        return self
    
    def generate_cuda_templates(self, output_dir: str) -> Dict[str, str]:
        """Generate CUDA template files"""
        
        output_path = Path(output_dir)
        generated_files = {}
        
        # CUDA kernel file
        cuda_content = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for element-wise addition
__global__ void cuda_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ result,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for ReLU activation
__global__ void cuda_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

// CUDA kernel for matrix multiplication (simplified)
__global__ void cuda_matmul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    const int M, const int N, const int K) {
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// Host functions
torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input b must be float32");
    
    auto result = torch::empty_like(a);
    const int size = a.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    cuda_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    return result;
}

torch::Tensor cuda_relu(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    
    cuda_relu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    
    return output;
}

torch::Tensor cuda_matmul(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b.device().is_cuda(), "Input b must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input b must be float32");
    
    const auto M = a.size(0);
    const auto K = a.size(1);
    const auto N = b.size(1);
    
    TORCH_CHECK(K == b.size(0), "Matrix dimensions don't match for multiplication");
    
    auto c = torch::zeros({M, N}, a.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((N + threads.x - 1) / threads.x,
                      (M + threads.y - 1) / threads.y);
    
    cuda_matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K
    );
    
    return c;
}
'''
        
        cuda_file = output_path / f"{self.extension_name}_cuda.cu"
        with open(cuda_file, 'w') as f:
            f.write(cuda_content)
        generated_files["cuda_file"] = str(cuda_file)
        
        # CUDA C++ binding file
        cuda_cpp_content = f'''
#include <torch/extension.h>

// CUDA function declarations
torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b);
torch::Tensor cuda_relu(torch::Tensor input);
torch::Tensor cuda_matmul(torch::Tensor a, torch::Tensor b);

// CPU implementations (fallback)
torch::Tensor cpu_add(torch::Tensor a, torch::Tensor b) {{
    return a + b;
}}

torch::Tensor cpu_relu(torch::Tensor input) {{
    return torch::clamp_min(input, 0);
}}

torch::Tensor cpu_matmul(torch::Tensor a, torch::Tensor b) {{
    return torch::matmul(a, b);
}}

// Dispatch functions
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {{
    if (a.device().is_cuda()) {{
        return cuda_add(a, b);
    }} else {{
        return cpu_add(a, b);
    }}
}}

torch::Tensor relu_forward(torch::Tensor input) {{
    if (input.device().is_cuda()) {{
        return cuda_relu(input);
    }} else {{
        return cpu_relu(input);
    }}
}}

torch::Tensor matmul_forward(torch::Tensor a, torch::Tensor b) {{
    if (a.device().is_cuda()) {{
        return cuda_matmul(a, b);
    }} else {{
        return cpu_matmul(a, b);
    }}
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("add", &add_forward, "Custom addition operation");
    m.def("relu", &relu_forward, "Custom ReLU activation");
    m.def("matmul", &matmul_forward, "Custom matrix multiplication");
}}
'''
        
        cuda_cpp_file = output_path / f"{self.extension_name}_cuda.cpp"
        with open(cuda_cpp_file, 'w') as f:
            f.write(cuda_cpp_content)
        generated_files["cuda_cpp_file"] = str(cuda_cpp_file)
        
        return generated_files

class CustomExtensionManager:
    """Manager for PyTorch custom extensions"""
    
    def __init__(self, base_dir: str = "./custom_extensions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.extensions = {}
    
    def create_cpp_extension(self, name: str) -> Dict[str, str]:
        """Create a new C++ extension"""
        
        extension_dir = self.base_dir / name
        extension_dir.mkdir(exist_ok=True)
        
        builder = CPPExtensionBuilder(name)
        
        # Generate template files
        generated_files = builder.generate_cpp_templates(str(extension_dir))
        
        # Add source files to builder
        builder.add_source_file(generated_files["cpp_file"])
        
        # Generate setup.py
        setup_file = builder.generate_setup_py(str(extension_dir))
        generated_files["setup_file"] = setup_file
        
        # Generate build script
        build_script = self._generate_build_script(name, str(extension_dir))
        generated_files["build_script"] = build_script
        
        self.extensions[name] = {
            "type": "cpp",
            "directory": str(extension_dir),
            "files": generated_files
        }
        
        print(f"✓ Created C++ extension '{name}' in {extension_dir}")
        return generated_files
    
    def create_cuda_extension(self, name: str) -> Dict[str, str]:
        """Create a new CUDA extension"""
        
        extension_dir = self.base_dir / name
        extension_dir.mkdir(exist_ok=True)
        
        builder = CUDAExtensionBuilder(name)
        
        # Generate template files
        cpp_files = builder.generate_cpp_templates(str(extension_dir))
        cuda_files = builder.generate_cuda_templates(str(extension_dir))
        
        generated_files = {**cpp_files, **cuda_files}
        
        # Add source files to builder
        builder.add_source_file(cuda_files["cuda_cpp_file"])
        builder.add_cuda_file(cuda_files["cuda_file"])
        
        # Generate setup.py for CUDA
        setup_content = self._generate_cuda_setup_py(name, generated_files)
        setup_file = extension_dir / "setup.py"
        with open(setup_file, 'w') as f:
            f.write(setup_content)
        generated_files["setup_file"] = str(setup_file)
        
        # Generate build script
        build_script = self._generate_build_script(name, str(extension_dir), is_cuda=True)
        generated_files["build_script"] = build_script
        
        self.extensions[name] = {
            "type": "cuda",
            "directory": str(extension_dir),
            "files": generated_files
        }
        
        print(f"✓ Created CUDA extension '{name}' in {extension_dir}")
        return generated_files
    
    def _generate_cuda_setup_py(self, name: str, files: Dict[str, str]) -> str:
        """Generate setup.py for CUDA extension"""
        
        return f'''
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='{name}',
    ext_modules=[
        CUDAExtension(
            name='{name}',
            sources=[
                '{files["cuda_cpp_file"]}',
                '{files["cuda_file"]}'
            ],
            extra_compile_args={{
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }}
        )
    ],
    cmdclass={{
        'build_ext': BuildExtension
    }}
)
'''
    
    def _generate_build_script(self, name: str, extension_dir: str, 
                             is_cuda: bool = False) -> str:
        """Generate build script for extension"""
        
        script_content = f'''#!/bin/bash
set -e

echo "Building {name} extension..."

cd {extension_dir}

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Install dependencies
echo "Installing build dependencies..."
pip install pybind11 torch

{"# Check CUDA availability" if is_cuda else ""}
{"python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"" if is_cuda else ""}

# Build extension
echo "Building extension..."
python setup.py build_ext --inplace

# Install extension
echo "Installing extension..."
pip install -e .

echo "✓ {name} extension built and installed successfully!"
'''
        
        script_file = Path(extension_dir) / "build.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        return str(script_file)
    
    def build_extension(self, name: str) -> bool:
        """Build an extension"""
        
        if name not in self.extensions:
            print(f"Extension '{name}' not found")
            return False
        
        extension_info = self.extensions[name]
        build_script = extension_info["files"]["build_script"]
        
        try:
            result = subprocess.run([build_script], capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                print(f"✓ Extension '{name}' built successfully")
                return True
            else:
                print(f"✗ Failed to build extension '{name}':")
                print(result.stderr)
                return False
        
        except Exception as e:
            print(f"✗ Error building extension '{name}': {e}")
            return False
    
    def create_autograd_function(self, name: str) -> str:
        """Create custom autograd function template"""
        
        autograd_content = f'''
import torch
from torch.autograd import Function

class {name}Function(Function):
    """Custom autograd function for {name} operation"""
    
    @staticmethod
    def forward(ctx, input1, input2):
        """Forward pass implementation"""
        # Save tensors for backward pass
        ctx.save_for_backward(input1, input2)
        
        # Implement your custom operation here
        # Example: element-wise multiplication
        result = input1 * input2
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass implementation"""
        # Retrieve saved tensors
        input1, input2 = ctx.saved_tensors
        
        # Compute gradients
        grad_input1 = grad_input2 = None
        
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output * input2
        
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_output * input1
        
        return grad_input1, grad_input2

# Convenience function
def {name.lower()}(input1, input2):
    """Apply {name} operation"""
    return {name}Function.apply(input1, input2)

# Example usage:
if __name__ == "__main__":
    # Create test tensors
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)
    
    # Apply custom operation
    z = {name.lower()}(x, y)
    
    # Backward pass
    loss = z.sum()
    loss.backward()
    
    print(f"Input shapes: {{x.shape}}, {{y.shape}}")
    print(f"Output shape: {{z.shape}}")
    print(f"Gradients computed: {{x.grad is not None}}, {{y.grad is not None}}")
'''
        
        autograd_file = self.base_dir / f"{name.lower()}_autograd.py"
        with open(autograd_file, 'w') as f:
            f.write(autograd_content)
        
        print(f"✓ Created autograd function template: {autograd_file}")
        return str(autograd_file)
    
    def list_extensions(self) -> Dict[str, Any]:
        """List all created extensions"""
        return self.extensions

class ExtensionBenchmark:
    """Benchmark custom extensions against PyTorch implementations"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_operation(self, custom_op: callable, pytorch_op: callable,
                          inputs: List[torch.Tensor], num_runs: int = 1000) -> Dict[str, float]:
        """Benchmark custom operation against PyTorch equivalent"""
        
        # Warmup
        for _ in range(10):
            _ = custom_op(*inputs)
            _ = pytorch_op(*inputs)
        
        # Synchronize if CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark custom operation
        custom_times = []
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
                result_custom = custom_op(*inputs)
                end_time.record()
                torch.cuda.synchronize()
                custom_times.append(start_time.elapsed_time(end_time))
            else:
                import time
                start = time.time()
                result_custom = custom_op(*inputs)
                end = time.time()
                custom_times.append((end - start) * 1000)  # Convert to ms
        
        # Benchmark PyTorch operation
        pytorch_times = []
        for _ in range(num_runs):
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
                result_pytorch = pytorch_op(*inputs)
                end_time.record()
                torch.cuda.synchronize()
                pytorch_times.append(start_time.elapsed_time(end_time))
            else:
                import time
                start = time.time()
                result_pytorch = pytorch_op(*inputs)
                end = time.time()
                pytorch_times.append((end - start) * 1000)
        
        # Calculate statistics
        custom_mean = np.mean(custom_times)
        pytorch_mean = np.mean(pytorch_times)
        
        return {
            "custom_mean_ms": custom_mean,
            "pytorch_mean_ms": pytorch_mean,
            "speedup": pytorch_mean / custom_mean,
            "custom_std_ms": np.std(custom_times),
            "pytorch_std_ms": np.std(pytorch_times),
            "num_runs": num_runs
        }

if __name__ == "__main__":
    print("Custom C++/CUDA Extensions for PyTorch")
    print("=" * 42)
    
    print("\n1. Extension Manager Setup")
    print("-" * 28)
    
    # Initialize extension manager
    manager = CustomExtensionManager("./demo_extensions")
    
    print("\n2. Creating C++ Extension")
    print("-" * 28)
    
    # Create C++ extension
    cpp_files = manager.create_cpp_extension("demo_cpp_ops")
    
    print("Generated files:")
    for file_type, filepath in cpp_files.items():
        print(f"  {file_type}: {filepath}")
    
    print("\n3. Creating CUDA Extension")
    print("-" * 29)
    
    # Create CUDA extension
    if torch.cuda.is_available():
        cuda_files = manager.create_cuda_extension("demo_cuda_ops")
        
        print("Generated CUDA files:")
        for file_type, filepath in cuda_files.items():
            print(f"  {file_type}: {filepath}")
    else:
        print("CUDA not available - skipping CUDA extension creation")
    
    print("\n4. Custom Autograd Function")
    print("-" * 31)
    
    # Create custom autograd function
    autograd_file = manager.create_autograd_function("CustomMultiply")
    
    # Demonstrate autograd function (simulate execution)
    print("Custom autograd function template created")
    print("Example usage in the generated file:")
    print("""
    x = torch.randn(3, 4, requires_grad=True)
    y = torch.randn(3, 4, requires_grad=True)
    z = custom_multiply(x, y)
    loss = z.sum()
    loss.backward()
    """)
    
    print("\n5. Extension Development Guide")
    print("-" * 33)
    
    development_steps = [
        "1. Define the mathematical operation",
        "2. Implement CPU version in C++",
        "3. Implement CUDA kernel (if GPU support needed)",
        "4. Add Python bindings with pybind11",
        "5. Handle tensor shapes and error checking",
        "6. Implement backward pass for autograd",
        "7. Write unit tests for correctness",
        "8. Benchmark against PyTorch equivalents",
        "9. Document the extension API",
        "10. Package and distribute"
    ]
    
    print("Extension Development Steps:")
    for step in development_steps:
        print(f"  {step}")
    
    print("\n6. Best Practices")
    print("-" * 19)
    
    best_practices = [
        "Use TORCH_CHECK for input validation",
        "Handle different tensor dtypes appropriately",
        "Ensure CUDA memory coalescing in kernels",
        "Implement both CPU and GPU versions",
        "Add comprehensive error handling",
        "Use at::parallel_for for CPU parallelization",
        "Profile and optimize memory access patterns",
        "Provide clear documentation and examples",
        "Write thorough unit tests",
        "Consider numerical stability",
        "Use appropriate CUDA grid/block dimensions",
        "Handle edge cases (empty tensors, etc.)"
    ]
    
    print("Extension Development Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n7. Common Use Cases")
    print("-" * 21)
    
    use_cases = {
        "Custom Activation Functions": "Specialized activations not in PyTorch",
        "Domain-Specific Operations": "Operations specific to your field",
        "Performance Optimization": "Faster implementations of existing ops",
        "Novel Algorithms": "Research algorithms not yet in PyTorch",
        "Hardware-Specific Code": "Leverage specific hardware features",
        "Fused Operations": "Combine multiple ops to reduce memory transfers",
        "Custom Loss Functions": "Complex loss functions with custom gradients",
        "Specialized Convolutions": "Custom convolution implementations"
    }
    
    print("Common Use Cases for Extensions:")
    for use_case, description in use_cases.items():
        print(f"  {use_case}: {description}")
    
    print("\n8. CUDA Development Tips")
    print("-" * 26)
    
    cuda_tips = [
        "Use shared memory for frequently accessed data",
        "Minimize divergent branches in warps",
        "Optimize memory access patterns (coalescing)",
        "Use appropriate block and grid dimensions",
        "Consider occupancy when choosing block size",
        "Use CUDA streams for overlapping computation",
        "Profile with NVIDIA Nsight for optimization",
        "Handle different tensor layouts (contiguous, etc.)",
        "Use atomic operations carefully (they serialize)",
        "Consider using cuBLAS/cuDNN when appropriate"
    ]
    
    print("CUDA Development Tips:")
    for i, tip in enumerate(cuda_tips, 1):
        print(f"{i:2d}. {tip}")
    
    print("\n9. Debugging Extensions")
    print("-" * 24)
    
    debugging_tips = [
        "Use TORCH_CHECK for runtime assertions",
        "Print tensor shapes and dtypes for debugging",
        "Use cuda-gdb for debugging CUDA kernels",
        "Add printf statements in CUDA kernels",
        "Use torch.autograd.gradcheck for gradient testing",
        "Test with different tensor sizes and dtypes",
        "Use address sanitizer for memory errors",
        "Check CUDA error codes after kernel launches",
        "Verify numerical results against reference implementations",
        "Use unit tests with known input/output pairs"
    ]
    
    print("Extension Debugging Tips:")
    for i, tip in enumerate(debugging_tips, 1):
        print(f"{i:2d}. {tip}")
    
    print("\n10. Build Instructions")
    print("-" * 23)
    
    build_instructions = {
        "Prerequisites": [
            "PyTorch >= 1.5.0",
            "CUDA toolkit (for CUDA extensions)",
            "Compatible C++ compiler",
            "pybind11 (pip install pybind11)"
        ],
        "Build Commands": [
            "cd extension_directory",
            "python setup.py build_ext --inplace",
            "pip install -e ."
        ],
        "Alternative (JIT)": [
            "from torch.utils.cpp_extension import load",
            "module = load(name='extension', sources=['extension.cpp'])"
        ]
    }
    
    for category, items in build_instructions.items():
        print(f"{category}:")
        for item in items:
            print(f"  - {item}")
        print()
    
    # List created extensions
    print("\n11. Created Extensions Summary")
    print("-" * 34)
    
    extensions = manager.list_extensions()
    
    if extensions:
        print("Created extensions:")
        for name, info in extensions.items():
            print(f"  {name} ({info['type']}): {info['directory']}")
    else:
        print("No extensions created in this session")
    
    print("\nTo build an extension:")
    print("  1. Navigate to the extension directory")
    print("  2. Run: bash build.sh")
    print("  3. Test: python -c \"import extension_name; print('Success!')\"")
    
    print("\nCustom C++/CUDA extensions demonstration completed!")
    print("Key components covered:")
    print("  - C++ and CUDA extension templates")
    print("  - Build system setup with pybind11")
    print("  - Custom autograd function creation")
    print("  - Best practices and debugging tips")
    print("  - CUDA kernel development guidelines")
    
    print("\nCustom extensions enable:")
    print("  - High-performance custom operations")
    print("  - Integration of existing C++/CUDA code")
    print("  - Novel algorithm implementations")
    print("  - Hardware-specific optimizations")
    print("  - Research prototyping and experimentation")