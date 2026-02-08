#!/usr/bin/env python3
"""PyTorch Random Tensor Generation - All random operations"""

import torch

print("=== Random Number Generation Basics ===")

# Set seed for reproducibility
torch.manual_seed(42)
print("Set manual seed to 42")

# Basic random tensors
rand_uniform = torch.rand(3, 4)  # Uniform [0, 1)
randn_normal = torch.randn(3, 4)  # Normal (0, 1)
print(f"Uniform [0,1): {rand_uniform.shape}")
print(f"Normal (0,1): {randn_normal.shape}")

# Random integers
randint_tensor = torch.randint(0, 10, (3, 4))  # Integers [0, 10)
randint_low_high = torch.randint(-5, 5, (2, 3))  # Integers [-5, 5)
print(f"Random int [0,10): {randint_tensor}")
print(f"Random int [-5,5): {randint_low_high}")

print("\n=== Random Distributions ===")

# Normal distribution with custom mean and std
normal_custom = torch.normal(mean=5.0, std=2.0, size=(3, 4))
print(f"Normal (mean=5, std=2): {normal_custom}")

# Normal with tensor parameters
means = torch.tensor([1.0, 2.0, 3.0])
stds = torch.tensor([0.5, 1.0, 1.5])
normal_tensor_params = torch.normal(means, stds)
print(f"Normal with tensor params: {normal_tensor_params}")

# Exponential distribution
exponential_tensor = torch.exponential(torch.ones(3, 4))
print(f"Exponential distribution shape: {exponential_tensor.shape}")

# Cauchy distribution
cauchy_tensor = torch.empty(3, 4).cauchy_(median=0.0, sigma=1.0)
print(f"Cauchy distribution shape: {cauchy_tensor.shape}")

# Log normal distribution
log_normal = torch.empty(3, 4).log_normal_(mean=0.0, std=1.0)
print(f"Log normal shape: {log_normal.shape}")

print("\n=== Random Sampling Functions ===")

# Random permutation
perm_10 = torch.randperm(10)
perm_dtype = torch.randperm(8, dtype=torch.float32)
print(f"Random permutation: {perm_10}")
print(f"Float permutation: {perm_dtype}")

# Multinomial sampling
weights = torch.tensor([0.1, 0.3, 0.4, 0.2])
samples = torch.multinomial(weights, num_samples=10, replacement=True)
print(f"Multinomial samples: {samples}")

# Poisson distribution
rates = torch.tensor([1.0, 2.0, 3.0, 4.0])
poisson_samples = torch.poisson(rates)
print(f"Poisson samples: {poisson_samples}")

print("\n=== Random Like Functions ===")

# Create random tensors with same shape as existing tensor
base_tensor = torch.zeros(2, 3, 4)

rand_like = torch.rand_like(base_tensor)
randn_like = torch.randn_like(base_tensor)
randint_like = torch.randint_like(base_tensor, 0, 10)

print(f"Base tensor shape: {base_tensor.shape}")
print(f"Rand like shape: {rand_like.shape}")
print(f"Randn like shape: {randn_like.shape}")
print(f"Randint like shape: {randint_like.shape}")

# Random like with different dtype
rand_like_float = torch.rand_like(base_tensor, dtype=torch.float64)
print(f"Rand like float64 dtype: {rand_like_float.dtype}")

print("\n=== Seed Management ===")

# Manual seed
torch.manual_seed(123)
tensor1 = torch.randn(2, 2)

torch.manual_seed(123)  # Reset to same seed
tensor2 = torch.randn(2, 2)

print("With same seed:")
print(f"Tensor 1:\n{tensor1}")
print(f"Tensor 2:\n{tensor2}")
print(f"Are equal: {torch.equal(tensor1, tensor2)}")

# Different seeds
torch.manual_seed(456)
tensor3 = torch.randn(2, 2)
print(f"Different seed tensor:\n{tensor3}")

# Get current seed state
initial_state = torch.get_rng_state()
tensor_a = torch.randn(2, 2)

# Reset to saved state
torch.set_rng_state(initial_state)
tensor_b = torch.randn(2, 2)

print(f"Restored state equal: {torch.equal(tensor_a, tensor_b)}")

print("\n=== CUDA Random Generation ===")

if torch.cuda.is_available():
    # CUDA manual seed
    torch.cuda.manual_seed(42)
    cuda_tensor1 = torch.randn(2, 2, device='cuda')
    
    torch.cuda.manual_seed(42)
    cuda_tensor2 = torch.randn(2, 2, device='cuda')
    
    print("CUDA random tensors equal:", torch.equal(cuda_tensor1, cuda_tensor2))
    
    # All CUDA devices seed
    torch.cuda.manual_seed_all(789)
    
    # CUDA random state
    cuda_state = torch.cuda.get_rng_state()
    cuda_tensor_c = torch.randn(2, 2, device='cuda')
    
    torch.cuda.set_rng_state(cuda_state)
    cuda_tensor_d = torch.randn(2, 2, device='cuda')
    
    print("CUDA state restored equal:", torch.equal(cuda_tensor_c, cuda_tensor_d))
else:
    print("CUDA not available for random generation testing")

print("\n=== In-place Random Operations ===")

# In-place random filling
inplace_tensor = torch.empty(3, 4)

# Fill with uniform random
inplace_tensor.uniform_(0, 1)
print(f"After uniform_ fill:\n{inplace_tensor}")

# Fill with normal random
inplace_tensor.normal_(mean=0, std=1)
print(f"After normal_ fill:\n{inplace_tensor}")

# Fill with exponential
inplace_tensor.exponential_(lambd=1.0)
print(f"After exponential_ fill:\n{inplace_tensor}")

# Fill with geometric
inplace_tensor.geometric_(p=0.5)
print(f"After geometric_ fill:\n{inplace_tensor}")

print("\n=== Random Initialization Patterns ===")

# Xavier/Glorot initialization
def xavier_uniform(tensor):
    fan_in = tensor.size(-1)
    fan_out = tensor.size(0) if tensor.dim() > 1 else tensor.size(0)
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    tensor.uniform_(-bound, bound)
    return tensor

xavier_tensor = xavier_uniform(torch.empty(4, 3))
print(f"Xavier uniform init:\n{xavier_tensor}")

# He initialization
def he_normal(tensor):
    fan_in = tensor.size(-1)
    std = (2.0 / fan_in) ** 0.5
    tensor.normal_(0, std)
    return tensor

he_tensor = he_normal(torch.empty(4, 3))
print(f"He normal init:\n{he_tensor}")

print("\n=== Random Sampling Utilities ===")

# Random choice equivalent (using multinomial)
choices = torch.tensor([10, 20, 30, 40, 50])
indices = torch.randint(0, len(choices), (5,))
random_choices = choices[indices]
print(f"Random choices: {random_choices}")

# Weighted random choice
weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
weighted_indices = torch.multinomial(weights, 3, replacement=False)
weighted_choices = choices[weighted_indices]
print(f"Weighted choices: {weighted_choices}")

# Random shuffle equivalent
shuffle_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
shuffled_indices = torch.randperm(len(shuffle_tensor))
shuffled = shuffle_tensor[shuffled_indices]
print(f"Shuffled: {shuffled}")

print("\n=== Random Tensor Properties ===")

# Check statistical properties
large_normal = torch.randn(10000)
print(f"Large normal tensor stats:")
print(f"  Mean: {large_normal.mean():.4f}")
print(f"  Std: {large_normal.std():.4f}")
print(f"  Min: {large_normal.min():.4f}")
print(f"  Max: {large_normal.max():.4f}")

large_uniform = torch.rand(10000)
print(f"Large uniform tensor stats:")
print(f"  Mean: {large_uniform.mean():.4f}")
print(f"  Std: {large_uniform.std():.4f}")
print(f"  Min: {large_uniform.min():.4f}")
print(f"  Max: {large_uniform.max():.4f}")

print("\n=== Random Number Generator ===")

# Create separate RNG
rng = torch.Generator()
rng.manual_seed(999)

# Use generator with random functions
gen_tensor1 = torch.randn(2, 2, generator=rng)
gen_tensor2 = torch.rand(2, 2, generator=rng)

print(f"Generator tensor 1:\n{gen_tensor1}")
print(f"Generator tensor 2:\n{gen_tensor2}")

# Reset generator
rng.manual_seed(999)
gen_tensor3 = torch.randn(2, 2, generator=rng)
print(f"Reset generator equal: {torch.equal(gen_tensor1, gen_tensor3)}")

print("\n=== Distribution Sampling ===")

# Beta distribution
alpha = torch.tensor([1.0, 2.0, 3.0])
beta = torch.tensor([1.0, 1.0, 1.0])
beta_samples = torch.distributions.Beta(alpha, beta).sample()
print(f"Beta samples: {beta_samples}")

# Gamma distribution
concentration = torch.tensor([1.0, 2.0, 3.0])
rate = torch.tensor([1.0, 1.0, 1.0])
gamma_samples = torch.distributions.Gamma(concentration, rate).sample()
print(f"Gamma samples: {gamma_samples}")

# Uniform distribution
low = torch.tensor([0.0, 1.0, 2.0])
high = torch.tensor([1.0, 2.0, 3.0])
uniform_samples = torch.distributions.Uniform(low, high).sample()
print(f"Uniform samples: {uniform_samples}")

print("\n=== Random Generation Complete ===") 