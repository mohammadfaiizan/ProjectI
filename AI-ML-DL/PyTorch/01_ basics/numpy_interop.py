"""
numpy_interop.py

Learn seamless interoperability between PyTorch tensors and NumPy arrays:
- Conversion between tensors and arrays
- Shared memory behavior
- Mixed operations
- Performance considerations
"""

import torch
import numpy as np

def main():
    # ================================================================== #
    #                     Basic Conversion                              #
    # ================================================================== #
    
    # Create NumPy array
    np_array = np.array([[1, 2], [3, 4]])
    print("Original NumPy array:\n", np_array)
    
    # Convert to PyTorch tensor (shares memory by default if on CPU)
    tensor_from_np = torch.from_numpy(np_array)
    print("\nTensor from NumPy:\n", tensor_from_np)
    
    # Convert back to NumPy array
    np_from_tensor = tensor_from_np.numpy()
    print("\nNumPy from tensor:\n", np_from_tensor)

    # ================================================================== #
    #                     Shared Memory Behavior                        #
    # ================================================================== #
    
    # Modify original array (changes will reflect in tensor)
    np_array[0, 0] = 100
    print("\nAfter modifying NumPy array:")
    print("NumPy array:\n", np_array)
    print("PyTorch tensor:\n", tensor_from_np)

    # Modify tensor (changes will reflect in NumPy array)
    tensor_from_np[1, 1] = 200
    print("\nAfter modifying tensor:")
    print("NumPy array:\n", np_array)
    print("PyTorch tensor:\n", tensor_from_np)

    # ================================================================== #
    #                     Breaking Memory Sharing                       #
    # ================================================================== #
    
    # Create copy to break memory sharing
    tensor_copy = torch.tensor(np_array)  # Creates new memory
    np_array[0, 0] = 0
    print("\nAfter breaking memory sharing:")
    print("NumPy array:\n", np_array)
    print("Tensor copy:\n", tensor_copy)

    # ================================================================== #
    #                     Device Considerations                         #
    # ================================================================== #
    
    # GPU tensors require explicit copy to CPU for conversion
    if torch.cuda.is_available():
        gpu_tensor = tensor_from_np.cuda()
        try:
            # This will fail - must move to CPU first
            # gpu_tensor.numpy()
            pass
        except RuntimeError as e:
            print("\nGPU conversion error:", e)
        
        # Proper conversion
        cpu_tensor = gpu_tensor.cpu()
        np_from_gpu = cpu_tensor.numpy()
        print("\nConversion from GPU tensor successful")

    # ================================================================== #
    #                     Mixed Operations                              #
    # ================================================================== #
    
    # Automatic conversion in operations
    mixed_add = tensor_from_np + np.array([10, 20])  # Tensor + NumPy array
    print("\nMixed addition result:\n", mixed_add)
    print("Result type:", type(mixed_add))  # Returns torch.Tensor

    # Using NumPy functions on tensors
    tensor = torch.randn(2, 3)
    np_sin = np.sin(tensor)  # Returns NumPy array
    torch_sin = torch.sin(tensor)  # Returns Tensor
    print("\nSin operations:")
    print("NumPy sin type:", type(np_sin))
    print("Torch sin type:", type(torch_sin))

    # ================================================================== #
    #                     Performance Benchmarks                        #
    # ================================================================== #
    
    # Large array/tensor operations
    large_np = np.random.rand(10000, 10000)
    large_tensor = torch.from_numpy(large_np)
    
    # Matrix multiplication comparison
    def benchmark(op, name):
        import time
        start = time.time()
        result = op()
        return f"{name}: {time.time() - start:.4f}s"
    
    print("\nPerformance comparison:")
    print(benchmark(lambda: np.matmul(large_np, large_np), "NumPy"))
    print(benchmark(lambda: torch.mm(large_tensor, large_tensor), "PyTorch CPU"))
    
    if torch.cuda.is_available():
        large_tensor_gpu = large_tensor.cuda()
        torch.cuda.synchronize()  # Wait for CUDA ops to finish
        print(benchmark(lambda: torch.mm(large_tensor_gpu, large_tensor_gpu), "PyTorch GPU"))

    # ================================================================== #
    #                     Type Conversions                              #
    # ================================================================== #
    
    # Handling different data types
    int_np = np.array([1, 2, 3], dtype=np.int32)
    float_tensor = torch.tensor([1., 2., 3.])
    
    # Conversion preserves types
    print("\nType preservation:")
    print("Tensor from int32 NumPy:", torch.from_numpy(int_np).dtype)
    print("NumPy from float32 tensor:", float_tensor.numpy().dtype)

    # ================================================================== #
    #                     Advanced Use Cases                            #
    # ================================================================== #
    
    # Using NumPy with autograd
    class HybridFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # Use NumPy in forward pass
            np_data = input.detach().numpy()
            result = torch.from_numpy(np_data ** 2).to(input.device)
            ctx.save_for_backward(input)
            return result
        
        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * 2 * input

    x = torch.tensor([2.0], requires_grad=True)
    y = HybridFunction.apply(x)
    y.backward()
    print("\nHybrid NumPy/PyTorch gradient:", x.grad)

if __name__ == "__main__":
    main()
    print("\nNumPy interoperability covered successfully!")
    print("Next: Explore data_loading/custom_dataset.py for data pipeline creation")