"""
tensors_operations.py

Fundamental operations with PyTorch tensors:
- Tensor creation
- Basic tensor operations
- Tensor indexing/slicing
- Tensor reshaping
- Tensor device management (CPU/GPU)
"""

import torch
import numpy as np

def main():
    # ================================================================== #
    #                      Tensor Creation Basics                       #
    # ================================================================== #
    
    # Create tensor from Python list
    data = [[1, 2], [3, 4]]
    x = torch.tensor(data)
    print("Tensor from list:\n", x)
    
    # Create special tensors
    zeros_tensor = torch.zeros(2, 3)        # 2x3 matrix of zeros
    ones_tensor = torch.ones(2, 3)          # 2x3 matrix of ones
    rand_tensor = torch.rand(2, 3)          # 2x3 matrix with uniform random [0,1)
    arange_tensor = torch.arange(0, 10, 2)  # Similar to range(0, 10, 2)
    eye_matrix = torch.eye(3)               # 3x3 identity matrix
    
    print("\nSpecial tensors:")
    print("Zeros tensor:\n", zeros_tensor)
    print("Random tensor:\n", rand_tensor)
    print("Arange tensor:", arange_tensor)
    print("Identity matrix:\n", eye_matrix)

    # ================================================================== #
    #                      Tensor Properties                            #
    # ================================================================== #
    
    tensor = torch.randn(3, 4)  # 3x4 matrix from normal distribution
    
    print("\nTensor properties:")
    print("Shape:", tensor.shape)       # torch.Size([3, 4])
    print("Data type:", tensor.dtype)   # torch.float32 (default)
    print("Device:", tensor.device)     # cpu (default)

    # ================================================================== #
    #                      Tensor Device Management                     #
    # ================================================================== #
    
    # Move tensor to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        gpu_tensor = tensor.to(device)
        print(f"\nTensor moved to {device}:")
        print(gpu_tensor.device)
    except RuntimeError as e:
        print("\nError moving tensor to GPU:", e)
    
    # Create tensor directly on target device
    device_tensor = torch.tensor([1, 2, 3], device=device)
    print("\nTensor created directly on", device_tensor.device)

    # ================================================================== #
    #                      Tensor Operations                            #
    # ================================================================== #
    
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    # Basic arithmetic operations
    add_result = a + b            # Element-wise addition
    sub_result = a - b            # Element-wise subtraction
    mul_result = a * b            # Element-wise multiplication
    div_result = b / a            # Element-wise division
    dot_product = torch.dot(a, b) # Dot product (1*4 + 2*5 + 3*6)
    
    print("\nArithmetic operations:")
    print("Addition:", add_result)
    print("Dot product:", dot_product.item())  # .item() for scalar value

    # Matrix multiplication
    mat1 = torch.randn(2, 3)
    mat2 = torch.randn(3, 4)
    matmul_result = torch.matmul(mat1, mat2)
    print("\nMatrix multiplication result shape:", matmul_result.shape)

    # In-place operations (modify tensor directly)
    a.add_(5)  # Adds 5 to all elements (note the _ suffix)
    print("\nAfter in-place addition:", a)

    # ================================================================== #
    #                      Tensor Reshaping                             #
    # ================================================================== #
    
    original = torch.arange(12)
    print("\nOriginal tensor:", original)
    
    # Reshape to 3x4 matrix
    reshaped = original.view(3, 4)  # view() requires contiguous memory
    print("Reshaped (3x4):\n", reshaped)
    
    # Flatten tensor
    flattened = reshaped.reshape(-1)  # Alternative: reshaped.flatten()
    print("Flattened:", flattened)
    
    # Add/remove dimensions
    unsqueezed = original.unsqueeze(0)  # Add batch dimension
    squeezed = unsqueezed.squeeze()     # Remove singleton dimensions
    print("\nUnsqueezed shape:", unsqueezed.shape)
    print("Squeezed shape:", squeezed.shape)

    # ================================================================== #
    #                      Tensor Indexing/Slicing                      #
    # ================================================================== #
    
    tensor_3d = torch.rand(2, 3, 4)  # 2 batches, 3 rows, 4 columns
    
    print("\n3D tensor shape:", tensor_3d.shape)
    print("First element:", tensor_3d[0].shape)       # 3x4
    print("First row of first element:", tensor_3d[0, 0].shape)  # 4 elements
    print("Last column:", tensor_3d[:, :, -1].shape)  # 2x3

    # ================================================================== #
    #                      Tensor Broadcasting                          #
    # ================================================================== #
    
    # Tensors of different shapes can be operated on through broadcasting
    matrix = torch.ones(3, 4)
    vector = torch.arange(1, 5)  # [1, 2, 3, 4]
    
    broadcast_result = matrix + vector  # Vector broadcast across matrix rows
    print("\nBroadcasting result shape:", broadcast_result.shape)

    # ================================================================== #
    #                      Tensor & NumPy Interop                       #
    # ================================================================== #
    
    # Tensor to NumPy array (shares memory if tensor is on CPU)
    numpy_array = tensor.numpy()
    print("\nTensor to NumPy array:", type(numpy_array))
    
    # NumPy array to Tensor
    new_tensor = torch.from_numpy(numpy_array)
    print("NumPy array to tensor:", type(new_tensor))

    # ================================================================== #
    #                      Advanced Operations                          #
    # ================================================================== #
    
    # Clipping values
    clipped = torch.clamp(tensor, min=-0.5, max=0.5)
    print("\nClipped tensor values between [-0.5, 0.5]")
    
    # Concatenation
    cat_tensor = torch.cat([a.unsqueeze(0), b.unsqueeze(0)], dim=0)
    print("\nConcatenated tensors shape:", cat_tensor.shape)

if __name__ == "__main__":
    main()
    print("\nAll tensor operations executed successfully!")
    print("Next: Explore autograd_basics.py for automatic differentiation")