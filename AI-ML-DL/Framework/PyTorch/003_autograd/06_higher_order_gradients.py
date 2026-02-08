#!/usr/bin/env python3
"""PyTorch Higher-Order Gradients - Second order derivatives, create_graph"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== Higher-Order Gradients Overview ===")

print("Higher-order gradients enable:")
print("1. Second-order optimization methods")
print("2. Meta-learning algorithms")
print("3. Adversarial training")
print("4. Hessian computation")
print("5. Jacobian analysis")

print("\n=== Basic Second-Order Gradients ===")

# Simple second-order derivative
x = torch.tensor(2.0, requires_grad=True)
y = x**4  # y = x^4

# First-order derivative
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"x = {x}")
print(f"y = x^4 = {y}")
print(f"dy/dx = 4x^3 = {dy_dx}")

# Second-order derivative
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx² = 12x² = {d2y_dx2}")
print(f"Expected at x=2: 12*4 = 48")

print("\n=== Vector Second-Order Gradients ===")

# Vector function second derivatives
x_vec = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y_vec = (x_vec**3).sum()  # y = x1³ + x2³ + x3³

# First-order gradients
grad_first = torch.autograd.grad(y_vec, x_vec, create_graph=True)[0]
print(f"x_vec = {x_vec}")
print(f"First-order gradient = 3x² = {grad_first}")

# Second-order gradients (diagonal Hessian)
grad_second = []
for i in range(len(grad_first)):
    grad2 = torch.autograd.grad(grad_first[i], x_vec, retain_graph=True)[0]
    grad_second.append(grad2[i])  # Diagonal elements

grad_second = torch.stack(grad_second)
print(f"Second-order gradients = 6x = {grad_second}")

print("\n=== Hessian Computation ===")

def compute_hessian(func, inputs):
    """Compute Hessian matrix for scalar function"""
    inputs = inputs.requires_grad_(True)
    
    # First-order gradients
    output = func(inputs)
    grad_outputs = torch.autograd.grad(output, inputs, create_graph=True)[0]
    
    # Second-order gradients (Hessian)
    hessian = torch.zeros(inputs.size(0), inputs.size(0))
    
    for i in range(inputs.size(0)):
        grad2 = torch.autograd.grad(
            grad_outputs[i], inputs, 
            retain_graph=True, create_graph=False
        )[0]
        hessian[i] = grad2
    
    return hessian

# Test Hessian computation
def quadratic_func(x):
    """Quadratic function: f(x) = x^T A x + b^T x + c"""
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, -1.0])
    c = 0.5
    return x @ A @ x + b @ x + c

x_hess = torch.tensor([1.0, 2.0])
hessian = compute_hessian(quadratic_func, x_hess)

print(f"Input: {x_hess}")
print(f"Function value: {quadratic_func(x_hess)}")
print(f"Hessian matrix:\n{hessian}")
print("Expected Hessian: 2*A =")
print(torch.tensor([[4.0, 2.0], [2.0, 6.0]]))

print("\n=== Jacobian Computation ===")

def compute_jacobian(func, inputs):
    """Compute Jacobian matrix for vector function"""
    inputs = inputs.requires_grad_(True)
    outputs = func(inputs)
    
    jacobian = torch.zeros(outputs.size(0), inputs.size(0))
    
    for i in range(outputs.size(0)):
        grad_outputs = torch.zeros_like(outputs)
        grad_outputs[i] = 1.0
        
        grads = torch.autograd.grad(
            outputs, inputs, grad_outputs=grad_outputs,
            retain_graph=True, create_graph=False
        )[0]
        
        jacobian[i] = grads
    
    return jacobian

# Test Jacobian computation
def vector_func(x):
    """Vector function: f(x) = [x1², x1*x2, x2³]"""
    return torch.stack([x[0]**2, x[0]*x[1], x[1]**3])

x_jac = torch.tensor([2.0, 3.0])
jacobian = compute_jacobian(vector_func, x_jac)

print(f"Input: {x_jac}")
print(f"Function output: {vector_func(x_jac)}")
print(f"Jacobian matrix:\n{jacobian}")

print("\n=== Higher-Order Gradients in Neural Networks ===")

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def compute_grad_grad(model, input_data, loss_fn):
    """Compute gradients of gradients (second-order)"""
    # First forward pass
    output = model(input_data)
    loss = loss_fn(output)
    
    # First-order gradients
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    # Flatten gradients
    flat_grads = torch.cat([g.view(-1) for g in grads])
    
    # Gradient of gradients (for demonstration, sum of squared gradients)
    grad_loss = (flat_grads**2).sum()
    
    # Second-order gradients
    grad_grads = torch.autograd.grad(grad_loss, model.parameters())
    
    return grads, grad_grads

# Test higher-order gradients in neural network
net = SimpleNet()
input_nn = torch.randn(4, 3, requires_grad=True)
loss_fn = lambda output: (output**2).sum()

first_grads, second_grads = compute_grad_grad(net, input_nn, loss_fn)

print("Neural network higher-order gradients:")
for i, (g1, g2) in enumerate(zip(first_grads, second_grads)):
    print(f"Parameter {i}: 1st grad norm = {g1.norm():.6f}, 2nd grad norm = {g2.norm():.6f}")

print("\n=== Meta-Learning with Higher-Order Gradients ===")

def maml_style_update(model, support_data, support_labels, query_data, query_labels, 
                     inner_lr=0.01, meta_lr=0.001):
    """MAML-style meta-learning update using higher-order gradients"""
    
    criterion = nn.MSELoss()
    
    # Save original parameters
    original_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Inner loop: adapt to support set
    support_output = model(support_data)
    support_loss = criterion(support_output, support_labels)
    
    # Compute gradients with create_graph=True for meta-learning
    grads = torch.autograd.grad(support_loss, model.parameters(), create_graph=True)
    
    # Update parameters (fast adaptation)
    adapted_params = {}
    for (name, param), grad in zip(model.named_parameters(), grads):
        adapted_params[name] = param - inner_lr * grad
    
    # Replace model parameters with adapted ones
    for name, param in model.named_parameters():
        param.data = adapted_params[name]
    
    # Outer loop: evaluate on query set
    query_output = model(query_data)
    query_loss = criterion(query_output, query_labels)
    
    # Meta-gradients (gradients of gradients)
    meta_grads = torch.autograd.grad(query_loss, original_params.values())
    
    # Meta-update (would normally use meta-optimizer)
    for (name, param), meta_grad in zip(model.named_parameters(), meta_grads):
        original_params[name] -= meta_lr * meta_grad
    
    # Restore original parameters (updated with meta-gradients)
    for name, param in model.named_parameters():
        param.data = original_params[name]
    
    return support_loss.item(), query_loss.item()

# Test MAML-style learning
meta_model = SimpleNet()
support_x = torch.randn(5, 3)
support_y = torch.randn(5, 1)
query_x = torch.randn(3, 3)
query_y = torch.randn(3, 1)

support_loss, query_loss = maml_style_update(
    meta_model, support_x, support_y, query_x, query_y
)

print(f"MAML-style update completed:")
print(f"  Support loss: {support_loss:.4f}")
print(f"  Query loss: {query_loss:.4f}")

print("\n=== Adversarial Training with Higher-Order Gradients ===")

def fgsm_attack_higher_order(model, data, target, epsilon=0.1):
    """FGSM attack using higher-order gradients"""
    data.requires_grad_(True)
    
    # Forward pass
    output = model(data)
    loss = F.cross_entropy(output, target)
    
    # First-order gradient w.r.t. input
    grad = torch.autograd.grad(loss, data, create_graph=True)[0]
    
    # Second-order term (gradient of gradient norm)
    grad_norm = grad.norm()
    second_order = torch.autograd.grad(grad_norm, data)[0]
    
    # Combined adversarial perturbation
    perturbation = epsilon * (grad.sign() + 0.1 * second_order.sign())
    
    # Create adversarial example
    adversarial_data = data + perturbation
    
    return adversarial_data.detach()

# Test adversarial attack
class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

adv_model = ClassificationNet()
adv_data = torch.randn(3, 10, requires_grad=True)
adv_target = torch.randint(0, 5, (3,))

adversarial_examples = fgsm_attack_higher_order(adv_model, adv_data, adv_target)

print("Adversarial attack with higher-order gradients:")
print(f"Original data shape: {adv_data.shape}")
print(f"Adversarial data shape: {adversarial_examples.shape}")
print(f"Perturbation norm: {(adversarial_examples - adv_data).norm():.4f}")

print("\n=== Curvature Analysis ===")

def compute_gauss_newton_approx(model, data, target):
    """Compute Gauss-Newton approximation to Hessian"""
    criterion = nn.MSELoss()
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Compute Jacobian of residuals
    residuals = output - target
    jacobians = []
    
    for i in range(residuals.size(0)):
        for j in range(residuals.size(1)):
            if residuals[i, j].requires_grad:
                J = torch.autograd.grad(
                    residuals[i, j], model.parameters(),
                    retain_graph=True, create_graph=False
                )
                jacobians.append(torch.cat([j.view(-1) for j in J]))
    
    if jacobians:
        J_matrix = torch.stack(jacobians)
        # Gauss-Newton approximation: J^T J
        gauss_newton = J_matrix.t() @ J_matrix
        return gauss_newton
    
    return None

# Test curvature analysis
curve_model = SimpleNet()
curve_data = torch.randn(2, 3)
curve_target = torch.randn(2, 1)

gn_approx = compute_gauss_newton_approx(curve_model, curve_data, curve_target)
if gn_approx is not None:
    print(f"Gauss-Newton approximation shape: {gn_approx.shape}")
    print(f"Condition number: {torch.linalg.cond(gn_approx):.2f}")

print("\n=== Efficient Higher-Order Gradient Computation ===")

def hvp(vector, model, inputs, targets, damping=0.01):
    """Hessian-vector product (more efficient than full Hessian)"""
    criterion = nn.MSELoss()
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # First-order gradients
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    
    # Gradient-vector product
    gv_product = (flat_grads * vector).sum()
    
    # Hessian-vector product
    hvp_result = torch.autograd.grad(gv_product, model.parameters())
    hvp_flat = torch.cat([h.view(-1) for h in hvp_result])
    
    # Add damping for numerical stability
    return hvp_flat + damping * vector

# Test HVP computation
hvp_model = SimpleNet()
hvp_inputs = torch.randn(4, 3)
hvp_targets = torch.randn(4, 1)

# Random vector for HVP
num_params = sum(p.numel() for p in hvp_model.parameters())
random_vector = torch.randn(num_params)

hvp_result = hvp(random_vector, hvp_model, hvp_inputs, hvp_targets)
print(f"Hessian-vector product computed:")
print(f"  Input vector norm: {random_vector.norm():.6f}")
print(f"  HVP result norm: {hvp_result.norm():.6f}")

print("\n=== K-FAC Approximation ===")

def kfac_approximation(model, data, target):
    """Kronecker-factored approximation for efficient second-order updates"""
    criterion = nn.MSELoss()
    
    # Store activations and gradients
    activations = {}
    gradients = {}
    
    def save_activation(name):
        def hook(module, input, output):
            activations[name] = input[0].detach()
        return hook
    
    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(save_activation(name)))
            hooks.append(module.register_backward_hook(save_gradient(name)))
    
    # Forward and backward pass
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    # Compute K-FAC matrices
    kfac_matrices = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in activations and name in gradients:
            # Input covariance
            a = activations[name]
            if len(a.shape) == 2:  # Add bias term
                a = torch.cat([a, torch.ones(a.size(0), 1)], dim=1)
            A = a.t() @ a / a.size(0)
            
            # Gradient covariance
            g = gradients[name]
            G = g.t() @ g / g.size(0)
            
            kfac_matrices[name] = {'A': A, 'G': G}
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    return kfac_matrices

# Test K-FAC approximation
kfac_model = SimpleNet()
kfac_data = torch.randn(8, 3)
kfac_target = torch.randn(8, 1)

kfac_mats = kfac_approximation(kfac_model, kfac_data, kfac_target)
print("K-FAC approximation computed:")
for name, matrices in kfac_mats.items():
    print(f"  Layer {name}:")
    print(f"    A matrix shape: {matrices['A'].shape}")
    print(f"    G matrix shape: {matrices['G'].shape}")

print("\n=== Higher-Order Gradient Best Practices ===")

print("Higher-Order Gradient Guidelines:")
print("1. Use create_graph=True for differentiating gradients")
print("2. Be careful with memory usage - gradients of gradients are expensive")
print("3. Consider approximations (HVP, K-FAC) for efficiency")
print("4. Use retain_graph=True when computing multiple derivatives")
print("5. Clear computation graphs when not needed")
print("6. Monitor gradient norms to avoid numerical issues")

print("\nApplications:")
print("- Second-order optimization (L-BFGS, Natural gradients)")
print("- Meta-learning (MAML, Reptile)")
print("- Adversarial training")
print("- Uncertainty quantification")
print("- Neural architecture search")
print("- Physics-informed neural networks")

print("\nPerformance Considerations:")
print("- Memory usage scales quadratically with parameters")
print("- Computation time increases significantly")
print("- Use approximations for large models")
print("- Consider checkpointing for memory efficiency")
print("- Profile memory usage during development")

print("\n=== Higher-Order Gradients Complete ===") 