# Custom Autograd and Higher Order

## Table of Contents

- [torch.autograd.Function (forward/backward/ctx)](#torchautogradfunction-forwardbackwardctx)
- [Custom Gradient Definitions](#custom-gradient-definitions)
- [Higher-Order Gradients (create_graph=True)](#higher-order-gradients-create_graphtrue)
- [Hessian Computation](#hessian-computation)
- [Functional API (jacobian, hessian, vjp, jvp)](#functional-api-jacobian-hessian-vjp-jvp)
- [Backward Hooks and Gradient Modification](#backward-hooks-and-gradient-modification)

---

## torch.autograd.Function (forward/backward/ctx)

Custom autograd operations are defined by subclassing **torch.autograd.Function**. You implement **forward** and **backward** static methods. Use **ctx** to save tensors and non-tensor data for the backward pass.

```python
import torch

class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.pow(2)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * input * grad_output
        return grad_input

square = SquareFunction.apply
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = square(x)
y.sum().backward()
print(x.grad)
```

**ctx methods:**
- `ctx.save_for_backward(*tensors)` - Save tensors for backward
- `ctx.saved_tensors` - Retrieve saved tensors in backward
- `ctx.needs_input_grad` - Boolean tuple indicating which inputs need gradients

---

## Custom Gradient Definitions

**Multi-input custom function:**

```python
class MultiplyAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, z):
        ctx.save_for_backward(x, y, z)
        return x * y + z

    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        grad_x = grad_output * y
        grad_y = grad_output * x
        grad_z = grad_output
        return grad_x, grad_y, grad_z

multiply_add = MultiplyAddFunction.apply
```

**Non-differentiable operations with custom gradients (Straight-Through Estimator):**

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

round_ste = RoundSTE.apply
```

**Return None** for non-differentiable parameters:

```python
def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_input, None, None
```

---

## Higher-Order Gradients (create_graph=True)

Set **create_graph=True** in `torch.autograd.grad` or `backward()` to build a graph of the gradient computation, enabling second-order derivatives.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**4

grad_first = torch.autograd.grad(y, x, create_graph=True)[0]
grad_second = torch.autograd.grad(grad_first, x)[0]

print(grad_first)
print(grad_second)
```

**Vector second-order gradients:**

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x**3).sum()

grad_first = torch.autograd.grad(y, x, create_graph=True)[0]
grad_second = torch.autograd.grad(grad_first, x, retain_graph=True)[0]
```

---

## Hessian Computation

The **Hessian** is the matrix of second partial derivatives. Compute it by differentiating the gradient with respect to each input.

```python
def compute_hessian(func, inputs):
    inputs = inputs.requires_grad_(True)
    output = func(inputs)
    grad_outputs = torch.autograd.grad(output, inputs, create_graph=True)[0]
    hessian = torch.zeros(inputs.size(0), inputs.size(0))

    for i in range(inputs.size(0)):
        grad2 = torch.autograd.grad(grad_outputs[i], inputs, retain_graph=True)[0]
        hessian[i] = grad2

    return hessian

def quadratic_func(x):
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    return x @ A @ x

x = torch.tensor([1.0, 2.0])
hessian = compute_hessian(quadratic_func, x)
```

**Hessian-vector product (HVP)** for efficiency:

```python
def hvp(vector, model, inputs, targets):
    criterion = torch.nn.MSELoss()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    gv_product = (flat_grads * vector).sum()
    hvp_result = torch.autograd.grad(gv_product, model.parameters())
    return torch.cat([h.view(-1) for h in hvp_result])
```

---

## Functional API (jacobian, hessian, vjp, jvp)

The **torch.func** module (PyTorch 1.13+) provides functional transforms for gradients.

**ft.grad** - Gradient of scalar function:

```python
import torch.func as ft

def f(x):
    return (x**2).sum()

grad_f = ft.grad(f)
gradient = grad_f(x)
```

**ft.jacrev / ft.jacfwd** - Jacobian (reverse/forward mode):

```python
def vector_func(x):
    return torch.stack([x[0]**2 + x[1], x[0]*x[1], x[1]**2])

jacrev_func = ft.jacrev(vector_func)
jacobian = jacrev_func(x)
```

**ft.hessian** - Hessian of scalar function:

```python
hess_func = ft.hessian(scalar_func)
hessian = hess_func(x)
```

**ft.vmap** - Vectorized map for batch processing:

```python
vmapped_func = ft.vmap(single_sample_func, in_dims=(None, 0))
batch_result = vmapped_func(params, batch_inputs)
```

**ft.vjp** - Vector-Jacobian product:

```python
output, vjp_fn = ft.vjp(func, *inputs)
gradient = vjp_fn(grad_output)
```

---

## Backward Hooks and Gradient Modification

**Tensor hooks** - Register on tensors to inspect or modify gradients during backward:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

def hook_fn(grad):
    return grad * 2

hook_handle = x.register_hook(hook_fn)
y = (x**2).sum()
y.backward()
hook_handle.remove()
```

**Module backward hooks** - Register on nn.Module:

```python
def module_backward_hook(module, grad_input, grad_output):
    print(f"Grad output shape: {grad_output[0].shape}")
    return grad_input

hook = model.linear.register_backward_hook(module_backward_hook)
```

**Gradient monitoring:**

```python
class GradientMonitor:
    def __init__(self):
        self.gradients = {}

    def register(self, name, tensor):
        def hook(grad):
            self.gradients[name] = grad.norm().item()
            return grad
        tensor.register_hook(hook)
```

**Gradient modification (clipping via hook):**

```python
def clip_hook(grad, max_norm=1.0):
    grad_norm = grad.norm()
    if grad_norm > max_norm:
        return grad * (max_norm / grad_norm)
    return grad

param.register_hook(lambda g: clip_hook(g, 1.0))
```

**Best practices:**
- Always remove hooks when done to avoid memory leaks
- Use `ctx.save_for_backward` for tensors; `ctx.attr = value` for non-tensors
- Return `None` for non-differentiable arguments in backward
- Test custom functions with `torch.autograd.gradcheck`
