#!/usr/bin/env python3
"""PyTorch Broadcasting Rules and Mechanics"""

import torch

print("=== Broadcasting Fundamentals ===")

# Broadcasting allows operations between tensors of different shapes
# Rules:
# 1. Start from the trailing dimension and work backwards
# 2. Two dimensions are compatible if they are equal, or one of them is 1
# 3. Missing dimensions are assumed to be 1

print("Broadcasting Rules:")
print("1. Compare dimensions from right to left")
print("2. Dimensions are compatible if equal or one is 1")
print("3. Missing dimensions are assumed to be 1")

print("\n=== Basic Broadcasting Examples ===")

# Scalar with tensor
scalar = 5
vector = torch.tensor([1, 2, 3, 4])
result_scalar_vector = scalar + vector

print(f"Scalar: {scalar}")
print(f"Vector: {vector}")
print(f"Scalar + Vector: {result_scalar_vector}")
print(f"Broadcast shape: scalar() + vector{vector.shape} -> {result_scalar_vector.shape}")

# Vector with matrix
vector_1d = torch.tensor([1, 2, 3])
matrix_2d = torch.tensor([[10, 20, 30], [40, 50, 60]])

result_vector_matrix = vector_1d + matrix_2d
print(f"\nVector (1D): {vector_1d}")
print(f"Matrix (2D):\n{matrix_2d}")
print(f"Vector + Matrix:\n{result_vector_matrix}")
print(f"Broadcast: {vector_1d.shape} + {matrix_2d.shape} -> {result_vector_matrix.shape}")

print("\n=== Dimension Expansion Examples ===")

# Different dimension expansions
tensor_1x1 = torch.tensor([[5]])
tensor_3x1 = torch.tensor([[1], [2], [3]])
tensor_1x4 = torch.tensor([[10, 20, 30, 40]])

print(f"Tensor 1x1: {tensor_1x1.shape}")
print(f"Tensor 3x1: {tensor_3x1.shape}")
print(f"Tensor 1x4: {tensor_1x4.shape}")

# 1x1 broadcasts to any shape
result_1x1_3x1 = tensor_1x1 + tensor_3x1
result_1x1_1x4 = tensor_1x1 + tensor_1x4

print(f"1x1 + 3x1 result shape: {result_1x1_3x1.shape}")
print(f"1x1 + 1x4 result shape: {result_1x1_1x4.shape}")

# 3x1 with 1x4 creates 3x4
result_3x1_1x4 = tensor_3x1 + tensor_1x4
print(f"3x1 + 1x4 result shape: {result_3x1_1x4.shape}")
print(f"Result 3x4:\n{result_3x1_1x4}")

print("\n=== Multi-dimensional Broadcasting ===")

# 3D broadcasting examples
tensor_2x1x3 = torch.randn(2, 1, 3)
tensor_1x4x1 = torch.randn(1, 4, 1)
tensor_1x1x3 = torch.randn(1, 1, 3)

print(f"Tensor A shape: {tensor_2x1x3.shape}")
print(f"Tensor B shape: {tensor_1x4x1.shape}")
print(f"Tensor C shape: {tensor_1x1x3.shape}")

# Broadcasting in 3D
result_3d = tensor_2x1x3 + tensor_1x4x1
print(f"A + B result shape: {result_3d.shape}")

# Triple broadcasting
result_triple = tensor_2x1x3 + tensor_1x4x1 + tensor_1x1x3
print(f"A + B + C result shape: {result_triple.shape}")

print("\n=== Broadcasting with Different Operations ===")

# Broadcasting works with all element-wise operations
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([10, 20])

print(f"Matrix A:\n{a}")
print(f"Vector B: {b}")

# Different operations
add_result = a + b
sub_result = a - b
mul_result = a * b
div_result = a / b
pow_result = a ** torch.tensor([2, 1])  # Different powers

print(f"A + B:\n{add_result}")
print(f"A - B:\n{sub_result}")
print(f"A * B:\n{mul_result}")
print(f"A / B:\n{div_result}")
print(f"A ** [2,1]:\n{pow_result}")

print("\n=== Explicit Broadcasting Functions ===")

# torch.broadcast_tensors() function
tensor_x = torch.tensor([1, 2, 3])
tensor_y = torch.tensor([[10], [20]])

broadcasted_x, broadcasted_y = torch.broadcast_tensors(tensor_x, tensor_y)

print(f"Original X: {tensor_x.shape}")
print(f"Original Y: {tensor_y.shape}")
print(f"Broadcasted X: {broadcasted_x.shape}")
print(f"Broadcasted Y: {broadcasted_y.shape}")
print(f"Broadcasted X:\n{broadcasted_x}")
print(f"Broadcasted Y:\n{broadcasted_y}")

# torch.broadcast_shapes() to check compatibility
try:
    broadcast_shape = torch.broadcast_shapes((3, 1), (1, 4), (1, 1))
    print(f"Broadcast shapes result: {broadcast_shape}")
except RuntimeError as e:
    print(f"Broadcasting error: {e}")

print("\n=== Broadcasting Edge Cases ===")

# Empty tensors
empty_tensor = torch.empty(0, 3)
regular_tensor = torch.tensor([[1, 2, 3]])

try:
    empty_broadcast = empty_tensor + regular_tensor
    print(f"Empty broadcast shape: {empty_broadcast.shape}")
except RuntimeError as e:
    print(f"Empty tensor broadcast error: {e}")

# Size-1 dimensions
size_1_tensor = torch.randn(3, 1, 4, 1)
other_tensor = torch.randn(1, 5, 1, 2)

broadcast_result = size_1_tensor + other_tensor
print(f"Size-1 broadcast: {size_1_tensor.shape} + {other_tensor.shape} -> {broadcast_result.shape}")

print("\n=== Broadcasting Failures ===")

# Incompatible shapes
incompatible_a = torch.randn(3, 4)
incompatible_b = torch.randn(2, 3)

try:
    failure_result = incompatible_a + incompatible_b
except RuntimeError as e:
    print(f"Broadcasting failure: {str(e)[:60]}...")

# Another incompatible case
incompatible_c = torch.randn(5, 3)
incompatible_d = torch.randn(4, 1)

try:
    failure_result2 = incompatible_c + incompatible_d
except RuntimeError as e:
    print(f"Second broadcasting failure: {str(e)[:60]}...")

print("\n=== Manual Broadcasting vs Automatic ===")

# Manual expansion
manual_a = torch.tensor([[1], [2], [3]])
manual_b = torch.tensor([10, 20])

# Manual expansion using expand
expanded_a = manual_a.expand(3, 2)
expanded_b = manual_b.expand(3, 2)
manual_result = expanded_a + expanded_b

print(f"Manual expansion result:\n{manual_result}")

# Automatic broadcasting (same result)
auto_result = manual_a + manual_b
print(f"Automatic broadcast result:\n{auto_result}")
print(f"Results equal: {torch.equal(manual_result, auto_result)}")

print("\n=== Broadcasting in Advanced Operations ===")

# Broadcasting in matrix operations
batch_matrices = torch.randn(5, 3, 4)  # Batch of matrices
vector_batch = torch.randn(5, 4, 1)    # Batch of vectors

# Matrix-vector multiplication with broadcasting
mv_result = torch.bmm(batch_matrices, vector_batch)
print(f"Batch matmul shape: {batch_matrices.shape} @ {vector_batch.shape} -> {mv_result.shape}")

# Broadcasting with comparison operations
comparison_a = torch.tensor([[1, 2, 3], [4, 5, 6]])
comparison_b = torch.tensor([2, 3, 4])

greater_result = comparison_a > comparison_b
equal_result = comparison_a == comparison_b

print(f"Greater than broadcast:\n{greater_result}")
print(f"Equal broadcast:\n{equal_result}")

print("\n=== Broadcasting Memory Considerations ===")

# Broadcasting doesn't create copies until operation
base = torch.tensor([1, 2, 3])
print(f"Base tensor memory ptr: {base.data_ptr()}")

# expand() creates a view (no memory copy)
expanded = base.expand(4, 3)
print(f"Expanded tensor memory ptr: {expanded.data_ptr()}")
print(f"Same memory: {base.data_ptr() == expanded.data_ptr()}")

# repeat() creates actual copies
repeated = base.repeat(4, 1)
print(f"Repeated tensor memory ptr: {repeated.data_ptr()}")
print(f"Different memory: {base.data_ptr() != repeated.data_ptr()}")

print("\n=== Broadcasting Performance ===")

import time

# Performance comparison
large_a = torch.randn(1000, 1000)
large_b = torch.randn(1000, 1)

# Broadcasting operation
start_time = time.time()
broadcast_op = large_a + large_b
broadcast_time = time.time() - start_time

# Manual expansion (less efficient)
start_time = time.time()
expanded_b = large_b.expand_as(large_a)
manual_op = large_a + expanded_b
manual_time = time.time() - start_time

print(f"Broadcasting time: {broadcast_time:.6f} seconds")
print(f"Manual expansion time: {manual_time:.6f} seconds")
print(f"Broadcasting faster: {manual_time > broadcast_time}")

print("\n=== Custom Broadcasting Logic ===")

def safe_broadcast_add(tensor_a, tensor_b):
    """Safe broadcasting addition with shape checking"""
    try:
        # Check if broadcasting is possible
        broadcast_shape = torch.broadcast_shapes(tensor_a.shape, tensor_b.shape)
        result = tensor_a + tensor_b
        return result, True
    except RuntimeError as e:
        return str(e), False

# Test safe broadcasting
test_a = torch.randn(3, 4)
test_b = torch.randn(4)
test_c = torch.randn(5)

result1, success1 = safe_broadcast_add(test_a, test_b)
result2, success2 = safe_broadcast_add(test_a, test_c)

print(f"Broadcast A{test_a.shape} + B{test_b.shape}: {'Success' if success1 else 'Failed'}")
print(f"Broadcast A{test_a.shape} + C{test_c.shape}: {'Success' if success2 else 'Failed'}")

print("\n=== Broadcasting Rules Summary ===")

print("Broadcasting Compatibility Rules:")
print("✓ (3, 4) + (4,) -> (3, 4)")
print("✓ (3, 1) + (1, 4) -> (3, 4)")
print("✓ (3, 1, 4) + (2, 1, 1) -> (3, 2, 4)")
print("✗ (3, 4) + (3,) -> Error (dimensions don't align)")
print("✗ (3, 4) + (2, 4) -> Error (3 != 2)")

print("\nBroadcasting Best Practices:")
print("1. Understand shape compatibility before operations")
print("2. Use broadcasting for memory efficiency")
print("3. Be careful with unintended broadcasting")
print("4. Test with torch.broadcast_shapes() for validation")
print("5. Consider using einsum for complex operations")

print("\n=== Broadcasting Complete ===") 