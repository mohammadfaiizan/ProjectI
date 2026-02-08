import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from typing import Dict, List, Tuple, Optional, Any

# Sample Models for LibTorch C++ Deployment
class LibTorchCompatibleCNN(nn.Module):
    """CNN designed for LibTorch C++ deployment"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class OptimizedResNet(nn.Module):
    """Optimized ResNet for C++ deployment"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        # Simplified ResNet architecture
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Simplified residual blocks
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        # For residual connections
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # First residual block
        identity = x
        out = self.layer1(x)
        out = out + identity
        out = self.relu(out)
        
        # Second residual block with downsampling
        identity = self.downsample(out)
        out = self.layer2(out)
        out = out + identity
        out = self.relu(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

# LibTorch Model Exporter
class LibTorchExporter:
    """Export PyTorch models for LibTorch C++ deployment"""
    
    def __init__(self, export_dir: str = "libtorch_models"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_traced_model(self, model: nn.Module, 
                           example_input: torch.Tensor,
                           model_name: str) -> str:
        """Export traced model for LibTorch"""
        
        model.eval()
        
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, example_input)
            
            # Save traced model
            export_path = os.path.join(self.export_dir, f"{model_name}_traced.pt")
            traced_model.save(export_path)
            
            print(f"✓ Traced model exported: {export_path}")
            return export_path
        
        except Exception as e:
            print(f"✗ Model tracing failed: {e}")
            return None
    
    def export_scripted_model(self, model: nn.Module, 
                             model_name: str) -> str:
        """Export scripted model for LibTorch"""
        
        model.eval()
        
        try:
            # Script the model
            scripted_model = torch.jit.script(model)
            
            # Save scripted model
            export_path = os.path.join(self.export_dir, f"{model_name}_scripted.pt")
            scripted_model.save(export_path)
            
            print(f"✓ Scripted model exported: {export_path}")
            return export_path
        
        except Exception as e:
            print(f"✗ Model scripting failed: {e}")
            return None
    
    def create_model_metadata(self, model_name: str,
                             input_shape: Tuple[int, ...],
                             output_shape: Tuple[int, ...],
                             preprocessing: Dict[str, Any] = None) -> str:
        """Create metadata file for C++ deployment"""
        
        if preprocessing is None:
            preprocessing = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "resize": [224, 224]
            }
        
        metadata = {
            "model_name": model_name,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "preprocessing": preprocessing,
            "data_type": "float32",
            "framework": "LibTorch",
            "version": "1.0"
        }
        
        metadata_path = os.path.join(self.export_dir, f"{model_name}_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata created: {metadata_path}")
        return metadata_path

# C++ Code Generator
class CppCodeGenerator:
    """Generate C++ code for LibTorch deployment"""
    
    @staticmethod
    def generate_inference_code(model_name: str, 
                               input_shape: Tuple[int, ...],
                               num_classes: int) -> str:
        """Generate C++ inference code"""
        
        cpp_code = f'''
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>

class {model_name}Inference {{
private:
    torch::jit::script::Module model;
    torch::Device device;
    
public:
    {model_name}Inference(const std::string& model_path, bool use_gpu = false)
        : device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {{
        
        try {{
            // Load the model
            model = torch::jit::load(model_path);
            model.to(device);
            model.eval();
            
            std::cout << "Model loaded successfully on " 
                      << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        }} catch (const std::exception& e) {{
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }}
    }}
    
    std::vector<float> preprocess(const cv::Mat& image) {{
        // Resize image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size({input_shape[3]}, {input_shape[2]}));
        
        // Convert BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        
        // Convert to float and normalize
        rgb.convertTo(rgb, CV_32F, 1.0/255.0);
        
        // Normalize with ImageNet stats
        std::vector<float> mean = {{0.485f, 0.456f, 0.406f}};
        std::vector<float> std = {{0.229f, 0.224f, 0.225f}};
        
        std::vector<float> input_data;
        input_data.reserve({input_shape[1]} * {input_shape[2]} * {input_shape[3]});
        
        // Convert HWC to CHW format
        for (int c = 0; c < 3; ++c) {{
            for (int h = 0; h < rgb.rows; ++h) {{
                for (int w = 0; w < rgb.cols; ++w) {{
                    float pixel = rgb.at<cv::Vec3f>(h, w)[c];
                    pixel = (pixel - mean[c]) / std[c];
                    input_data.push_back(pixel);
                }}
            }}
        }}
        
        return input_data;
    }}
    
    std::vector<float> predict(const cv::Mat& image) {{
        // Preprocess image
        auto input_data = preprocess(image);
        
        // Create tensor
        auto tensor = torch::from_blob(
            input_data.data(),
            {{{input_shape[0]}, {input_shape[1]}, {input_shape[2]}, {input_shape[3]}}},
            torch::kFloat
        ).to(device);
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        torch::NoGradGuard no_grad;
        auto output = model.forward(inputs).toTensor();
        
        // Apply softmax
        output = torch::softmax(output, 1);
        
        // Convert to CPU and get data
        output = output.to(torch::kCPU);
        auto output_accessor = output.accessor<float, 2>();
        
        std::vector<float> result;
        for (int i = 0; i < {num_classes}; ++i) {{
            result.push_back(output_accessor[0][i]);
        }}
        
        return result;
    }}
    
    int predict_class(const cv::Mat& image) {{
        auto probabilities = predict(image);
        
        int predicted_class = 0;
        float max_prob = probabilities[0];
        
        for (int i = 1; i < probabilities.size(); ++i) {{
            if (probabilities[i] > max_prob) {{
                max_prob = probabilities[i];
                predicted_class = i;
            }}
        }}
        
        return predicted_class;
    }}
}};

// Example usage
int main(int argc, char* argv[]) {{
    if (argc != 3) {{
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return -1;
    }}
    
    try {{
        // Initialize model
        {model_name}Inference inference(argv[1], true); // Use GPU if available
        
        // Load image
        cv::Mat image = cv::imread(argv[2]);
        if (image.empty()) {{
            std::cerr << "Error: Could not load image " << argv[2] << std::endl;
            return -1;
        }}
        
        // Run inference
        auto start = std::chrono::high_resolution_clock::now();
        auto probabilities = inference.predict(image);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Print results
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        std::cout << "Predictions:" << std::endl;
        
        for (int i = 0; i < probabilities.size(); ++i) {{
            std::cout << "  Class " << i << ": " << probabilities[i] << std::endl;
        }}
        
        int predicted_class = inference.predict_class(image);
        std::cout << "Predicted class: " << predicted_class << std::endl;
        
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }}
    
    return 0;
}}
'''
        return cpp_code
    
    @staticmethod
    def generate_cmake_file(model_name: str) -> str:
        """Generate CMakeLists.txt for building C++ application"""
        
        cmake_content = f'''
cmake_minimum_required(VERSION 3.12)
project({model_name}_inference)

set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Add executable
add_executable({model_name}_inference main.cpp)

# Link libraries
target_link_libraries({model_name}_inference "${{TORCH_LIBRARIES}}")
target_link_libraries({model_name}_inference "${{OpenCV_LIBS}}")

# Set C++14 flag
set_property(TARGET {model_name}_inference PROPERTY CXX_STANDARD 14)

# Include directories
target_include_directories({model_name}_inference PRIVATE "${{OpenCV_INCLUDE_DIRS}}")

# Copy model file to build directory
configure_file(${{CMAKE_SOURCE_DIR}}/{model_name}_traced.pt 
               ${{CMAKE_BINARY_DIR}}/{model_name}_traced.pt COPYONLY)
'''
        return cmake_content
    
    @staticmethod
    def generate_build_script() -> str:
        """Generate build script for C++ application"""
        
        build_script = '''#!/bin/bash

# Build script for LibTorch C++ application

# Check if LibTorch is installed
if [ ! -d "libtorch" ]; then
    echo "Downloading LibTorch..."
    
    # Download appropriate LibTorch version
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA detected, downloading CUDA version..."
        wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu117.zip -O libtorch.zip
    else
        echo "CPU only version..."
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip -O libtorch.zip
    fi
    
    unzip libtorch.zip
    rm libtorch.zip
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH=../libtorch ..

# Build
make -j$(nproc)

echo "Build completed. Run with: ./build/model_inference model.pt image.jpg"
'''
        return build_script

# Performance Benchmark for C++ Deployment
class LibTorchBenchmark:
    """Benchmark LibTorch models for C++ deployment"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_model_export(self, model: nn.Module, 
                              example_input: torch.Tensor,
                              model_name: str) -> Dict[str, Any]:
        """Benchmark different export methods"""
        
        import time
        
        results = {}
        
        # Benchmark tracing
        start_time = time.time()
        try:
            traced_model = torch.jit.trace(model, example_input)
            trace_time = time.time() - start_time
            results['tracing'] = {
                'success': True,
                'time_seconds': trace_time
            }
        except Exception as e:
            results['tracing'] = {
                'success': False,
                'error': str(e)
            }
        
        # Benchmark scripting
        start_time = time.time()
        try:
            scripted_model = torch.jit.script(model)
            script_time = time.time() - start_time
            results['scripting'] = {
                'success': True,
                'time_seconds': script_time
            }
        except Exception as e:
            results['scripting'] = {
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def benchmark_inference_speed(self, model_path: str,
                                 input_shape: Tuple[int, ...],
                                 num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed of exported model"""
        
        import time
        
        # Load model
        try:
            model = torch.jit.load(model_path)
            model.eval()
        except Exception as e:
            return {'error': str(e)}
        
        # Create test input
        test_input = torch.randn(input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(test_input)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_time_ms': sum(times) / len(times),
            'min_time_ms': min(times),
            'max_time_ms': max(times),
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }

# Deployment Helper
class LibTorchDeploymentHelper:
    """Helper utilities for LibTorch deployment"""
    
    @staticmethod
    def create_deployment_package(model_path: str, 
                                 model_name: str,
                                 include_samples: bool = True) -> str:
        """Create complete deployment package"""
        
        import zipfile
        import tempfile
        
        # Create temporary directory for package
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = os.path.join(temp_dir, f"{model_name}_deployment")
            os.makedirs(package_dir, exist_ok=True)
            
            # Copy model file
            if os.path.exists(model_path):
                shutil.copy(model_path, package_dir)
            
            # Generate C++ code
            generator = CppCodeGenerator()
            
            # Main C++ file
            cpp_code = generator.generate_inference_code(model_name, (1, 3, 224, 224), 10)
            with open(os.path.join(package_dir, "main.cpp"), 'w') as f:
                f.write(cpp_code)
            
            # CMakeLists.txt
            cmake_content = generator.generate_cmake_file(model_name)
            with open(os.path.join(package_dir, "CMakeLists.txt"), 'w') as f:
                f.write(cmake_content)
            
            # Build script
            build_script = generator.generate_build_script()
            with open(os.path.join(package_dir, "build.sh"), 'w') as f:
                f.write(build_script)
            
            # Make build script executable
            os.chmod(os.path.join(package_dir, "build.sh"), 0o755)
            
            # README
            readme_content = f'''
# {model_name} LibTorch Deployment

## Requirements
- CMake 3.12+
- OpenCV
- LibTorch (will be downloaded by build script)

## Build Instructions
1. Run the build script: `./build.sh`
2. This will download LibTorch and build the application

## Usage
```bash
./build/{model_name}_inference model.pt image.jpg
```

## Files
- `main.cpp`: C++ inference code
- `CMakeLists.txt`: CMake configuration
- `build.sh`: Build script
- `{os.path.basename(model_path)}`: PyTorch model file

## Notes
- The application expects input images in standard formats (jpg, png, etc.)
- Input images are automatically resized to 224x224
- GPU acceleration is used if available
'''
            
            with open(os.path.join(package_dir, "README.md"), 'w') as f:
                f.write(readme_content)
            
            # Create zip package
            package_path = f"{model_name}_libtorch_deployment.zip"
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            print(f"✓ Deployment package created: {package_path}")
            return package_path

if __name__ == "__main__":
    print("LibTorch C++ Deployment")
    print("=" * 27)
    
    # Create sample models
    cnn_model = LibTorchCompatibleCNN(num_classes=10)
    resnet_model = OptimizedResNet(num_classes=10)
    
    example_input = torch.randn(1, 3, 224, 224)
    
    print("\n1. Model Export for LibTorch")
    print("-" * 33)
    
    exporter = LibTorchExporter("demo_libtorch_models")
    
    # Export traced models
    cnn_traced_path = exporter.export_traced_model(cnn_model, example_input, "cnn_model")
    resnet_traced_path = exporter.export_traced_model(resnet_model, example_input, "resnet_model")
    
    # Export scripted models
    cnn_scripted_path = exporter.export_scripted_model(cnn_model, "cnn_model")
    resnet_scripted_path = exporter.export_scripted_model(resnet_model, "resnet_model")
    
    # Create metadata
    cnn_metadata = exporter.create_model_metadata(
        "cnn_model", (1, 3, 224, 224), (1, 10)
    )
    
    print("\n2. C++ Code Generation")
    print("-" * 27)
    
    generator = CppCodeGenerator()
    
    # Generate C++ inference code
    cpp_code = generator.generate_inference_code("CNNModel", (1, 3, 224, 224), 10)
    
    # Save generated code
    with open("demo_libtorch_models/cnn_inference.cpp", 'w') as f:
        f.write(cpp_code)
    
    # Generate CMake file
    cmake_content = generator.generate_cmake_file("cnn_model")
    with open("demo_libtorch_models/CMakeLists.txt", 'w') as f:
        f.write(cmake_content)
    
    # Generate build script
    build_script = generator.generate_build_script()
    with open("demo_libtorch_models/build.sh", 'w') as f:
        f.write(build_script)
    
    print("✓ C++ inference code generated")
    print("✓ CMakeLists.txt generated")
    print("✓ Build script generated")
    
    print("\n3. Performance Benchmarking")
    print("-" * 32)
    
    benchmark = LibTorchBenchmark()
    
    # Benchmark export methods
    export_results = benchmark.benchmark_model_export(cnn_model, example_input, "cnn_model")
    
    print("Export Benchmark Results:")
    for method, result in export_results.items():
        if result['success']:
            print(f"  {method.capitalize()}: {result['time_seconds']:.4f} seconds")
        else:
            print(f"  {method.capitalize()}: Failed - {result['error']}")
    
    # Benchmark inference speed
    if cnn_traced_path:
        inference_results = benchmark.benchmark_inference_speed(cnn_traced_path, (1, 3, 224, 224))
        
        if 'error' not in inference_results:
            print(f"\nInference Benchmark (Traced CNN):")
            print(f"  Mean time: {inference_results['mean_time_ms']:.2f} ms")
            print(f"  Min time: {inference_results['min_time_ms']:.2f} ms")
            print(f"  Max time: {inference_results['max_time_ms']:.2f} ms")
        else:
            print(f"Inference benchmark failed: {inference_results['error']}")
    
    print("\n4. Deployment Package Creation")
    print("-" * 37)
    
    helper = LibTorchDeploymentHelper()
    
    # Create deployment package
    if cnn_traced_path:
        package_path = helper.create_deployment_package(cnn_traced_path, "cnn_model")
    
    print("\n5. LibTorch Best Practices")
    print("-" * 32)
    
    best_practices = [
        "Use torch.jit.trace for models without control flow",
        "Use torch.jit.script for models with control flow",
        "Test exported models thoroughly before deployment",
        "Use appropriate data types (float32 is standard)",
        "Handle exceptions in model loading and inference",
        "Implement proper preprocessing in C++",
        "Use GPU acceleration when available",
        "Profile memory usage in production",
        "Implement proper error handling",
        "Use OpenMP for CPU parallelization",
        "Consider batch processing for throughput",
        "Monitor performance in production"
    ]
    
    print("LibTorch Deployment Best Practices:")
    for i, practice in enumerate(best_practices, 1):
        print(f"{i:2d}. {practice}")
    
    print("\n6. C++ Build Instructions")
    print("-" * 31)
    
    build_instructions = '''
# Building LibTorch C++ Application

## Prerequisites:
1. Install CMake (3.12+)
2. Install OpenCV
3. Download LibTorch

## Steps:
1. Extract the deployment package
2. Run the build script: ./build.sh
3. Or manually:
   - mkdir build && cd build
   - cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
   - make

## Usage:
./cnn_model_inference model.pt image.jpg
'''
    
    print(build_instructions)
    
    print("\n7. Performance Optimization Tips")
    print("-" * 38)
    
    optimization_tips = [
        "Enable optimizations: torch.jit.optimize_for_inference()",
        "Use appropriate tensor memory layout",
        "Minimize memory allocations in inference loop",
        "Use tensor.to(device, non_blocking=True) for async transfers",
        "Implement proper batching for throughput",
        "Profile with torch.profiler in Python first",
        "Use Intel MKL-DNN for CPU optimization",
        "Consider TensorRT for NVIDIA GPU optimization",
        "Use mixed precision when appropriate",
        "Optimize preprocessing pipeline"
    ]
    
    print("Performance Optimization Tips:")
    for i, tip in enumerate(optimization_tips, 1):
        print(f"{i:2d}. {tip}")
    
    print("\n8. Troubleshooting Common Issues")
    print("-" * 38)
    
    troubleshooting = [
        "Model loading fails: Check LibTorch version compatibility",
        "Undefined symbols: Ensure proper linking of LibTorch libraries",
        "CUDA errors: Verify CUDA version matches LibTorch build",
        "Memory errors: Check tensor memory layout and device placement",
        "Performance issues: Profile and optimize preprocessing",
        "Build errors: Verify CMake configuration and dependencies"
    ]
    
    print("Common Issues and Solutions:")
    for i, issue in enumerate(troubleshooting, 1):
        print(f"{i}. {issue}")
    
    print("\nLibTorch C++ deployment demonstration completed!")
    print("Generated files:")
    print("  - demo_libtorch_models/ (exported models and C++ code)")
    print("  - cnn_model_libtorch_deployment.zip (complete deployment package)")
    
    print("\nNext steps:")
    print("1. Extract the deployment package")
    print("2. Install LibTorch and OpenCV")
    print("3. Run ./build.sh to compile")
    print("4. Test with: ./build/cnn_model_inference model.pt image.jpg")