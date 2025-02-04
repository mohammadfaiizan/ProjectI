"""
autograd_basics.py

Learn PyTorch's automatic differentiation system:
- Basic gradient calculation
- Backpropagation
- Custom gradients
- Gradient accumulation control
"""

import torch

def main():
    # ================================================================== #
    #                     Basic Gradient Calculation                     #
    # ================================================================== #
    
    # Create tensor with gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    
    # Simple computation
    y = x ** 2
    z = 3 * y + 1
    
    # Backward pass (gradient calculation)
    z.backward()  # dz/dx
    
    print("\nBasic gradient calculation:")
    print(f"x = {x.item()}, z = {z.item()}")
    print(f"dz/dx (analytical: 6x = 12): {x.grad.item()}")

    # ================================================================== #
    #                     Multi-variable Backpropagation                 #
    # ================================================================== #
    
    # Create multiple variables with gradients
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    
    # More complex computation
    c = a * b
    d = torch.sin(c)
    e = d + a ** 2
    
    # Compute gradients for multiple variables
    e.backward()
    
    print("\nMulti-variable gradients:")
    print(f"da: {a.grad.item()} (analytical: b*cos(ab) + 2a = 2*cos(2) + 2 ≈ {2*torch.cos(torch.tensor(2.0)).item() + 2})")
    print(f"db: {b.grad.item()} (analytical: a*cos(ab) = cos(2) ≈ {torch.cos(torch.tensor(2.0)).item()})")

    # ================================================================== #
    #                     Gradient Accumulation Control                 #
    # ================================================================== #
    
    # Manual gradient zeroing (important for training loops)
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    
    for epoch in range(3):
        output = (w * 3).sum()
        output.backward()
        print(f"\nEpoch {epoch+1} gradients:", w.grad)
        # w.grad.zero_()  # Uncomment to prevent gradient accumulation
    
    print("\nNote: Gradients accumulate across backward calls!")
    print("Always use optimizer.zero_grad() or tensor.grad.zero_() in training loops")

    # ================================================================== #
    #                     Custom Gradients with Hooks                   #
    # ================================================================== #
    
    # Create tensor with custom gradient behavior
    v = torch.tensor(3.0, requires_grad=True)
    
    # Define custom forward and backward
    class CustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input ** 2
        
        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * (2 * input + 1)  # Custom derivative
    
    # Apply custom function
    result = CustomFunction.apply(v)
    result.backward()
    
    print("\nCustom gradient calculation:")
    print(f"Original derivative (2x): {2*v.item()}")
    print(f"Custom derivative (2x+1): {v.grad.item()}")

    # ================================================================== #
    #                     Intermediate Gradients                       #
    # ================================================================== #
    
    # Access intermediate gradients using retain_grad()
    x1 = torch.tensor(1.0, requires_grad=True)
    x2 = torch.tensor(2.0, requires_grad=True)
    
    h = x1 * x2
    h.retain_grad()  # Preserve gradient for non-leaf tensor
    
    y = h ** 2
    y.backward()
    
    print("\nIntermediate gradients:")
    print(f"dh/dx1: {x1.grad} (analytical: x2 = 2)")
    print(f"dh/dx2: {x2.grad} (analytical: x1 = 1)")
    print(f"dy/dh: {h.grad} (analytical: 2h = 4)")

    # ================================================================== #
    #                     Disabling Gradient Tracking                   #
    # ================================================================== #
    
    # Context manager for disabling gradients
    with torch.no_grad():
        no_grad_tensor = torch.tensor(5.0)
        no_grad_op = no_grad_tensor ** 2
        print("\nOperations in no_grad context:")
        print(f"Gradient tracking: {no_grad_op.requires_grad}")

    # Detach tensors from computation graph
    original = torch.tensor(3.0, requires_grad=True)
    detached = original.detach()
    print(f"\nDetached tensor gradient tracking: {detached.requires_grad}")

    # ================================================================== #
    #                     Gradient Checking                            #
    # ================================================================== #
    
    # Numerical gradient verification
    def numerical_derivative(f, x, eps=1e-5):
        return (f(x + eps) - f(x - eps)) / (2 * eps)
    
    x = torch.tensor(2.0, requires_grad=True)
    f = lambda x: torch.sin(x ** 2)
    analytic = torch.autograd.grad(f(x), x)[0]
    numeric = numerical_derivative(f, x)
    
    print("\nGradient verification:")
    print(f"Analytic gradient: {analytic.item():.6f}")
    print(f"Numerical gradient: {numeric.item():.6f}")
    print(f"Difference: {abs(analytic - numeric).item():.2e}")

if __name__ == "__main__":
    main()
    print("\nAutograd basics covered successfully!")
    print("Next: Explore neural_networks/linear_regression.py for practical applications")