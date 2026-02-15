# Autograd Fundamentals and Gradient Computation

## Table of Contents

- [Computation Graph (Dynamic DAG)](#computation-graph-dynamic-dag)
- [requires_grad](#requires_grad)
- [Leaf vs Non-Leaf Tensors](#leaf-vs-non-leaf-tensors)
- [backward() Mechanics](#backward-mechanics)
- [Gradient Computation Rules](#gradient-computation-rules)
- [retain_graph](#retain_graph)
- [Gradient Tracking Enable/Disable](#gradient-tracking-enabledisable)
- [.grad Attribute](#grad-attribute)
- [.grad_fn Chain](#grad_fn-chain)

---

## Computation Graph (Dynamic DAG)

PyTorch uses **automatic differentiation (autograd)** to compute gradients. The core mechanism is a **dynamic** **Directed Acyclic Graph (DAG)** that records operations as they execute during the forward pass.

**Key concepts:**
- The graph is built **on-the-fly** during the forward pass
- Each operation creates a node in the graph
- Nodes are connected by edges representing data flow
- The graph is **released** after `backward()` unless `retain_graph=True`

```python
import torch

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a + b
d = c * 3
e = torch.sin(d)

print(f"c.grad_fn: {c.grad_fn}")
print(f"d.grad_fn: {d.grad_fn}")
print(f"e.grad_fn: {e.grad_fn}")

e.backward()
print(f"de/da = {a.grad}")
print(f"de/db = {b.grad}")
```

---

## requires_grad

The **requires_grad** attribute controls whether a tensor participates in gradient computation. When `True`, gradients are tracked for that tensor and all operations involving it.

**Setting requires_grad:**

```python
tensor1 = torch.randn(3, 3)
tensor1.requires_grad_(True)

tensor2 = torch.randn(3, 3).requires_grad_(True)

tensor3 = torch.randn(3, 3, requires_grad=True)  # At creation
```

**Inheritance:** Operations that mix `requires_grad=True` and `requires_grad=False` tensors produce outputs with `requires_grad=True` if any input has it.

```python
x = torch.tensor([1.0, 2.0])
y = torch.tensor([3.0, 4.0], requires_grad=True)
z = x + y
print(z.requires_grad)
```

---

## Leaf vs Non-Leaf Tensors

**Leaf tensors** are tensors created directly by the user (e.g., `torch.tensor()`, `torch.randn`). **Non-leaf tensors** are created by operations on other tensors.

| Property | Leaf | Non-Leaf |
|----------|------|----------|
| Creation | User-created | Operation result |
| is_leaf | True | False |
| grad_fn | None | Backward function |
| grad retention | Default | Requires retain_grad() |

```python
leaf_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
non_leaf = leaf_tensor + 1
another_non_leaf = non_leaf * 2

print(leaf_tensor.is_leaf)
print(non_leaf.is_leaf)

loss = another_non_leaf.sum()
loss.backward()

print(leaf_tensor.grad)
print(non_leaf.grad)
```

**Model parameters** are leaf tensors by default.

---

## backward() Mechanics

The **backward()** method computes gradients by traversing the computation graph in reverse. It propagates gradients from the output tensor back to leaf tensors.

**For scalar outputs:**

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)
```

**For non-scalar outputs**, provide a `grad_output` tensor (Jacobian-vector product):

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2
grad_output = torch.tensor([1.0, 1.0])
y.backward(grad_output)
print(x.grad)
```

**Manual gradient computation** with `torch.autograd.grad`:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x**2).sum()
grads = torch.autograd.grad(y, x)
print(grads[0])
```

---

## Gradient Computation Rules

PyTorch computes gradients using the chain rule. Each operation has a corresponding backward function that defines how gradients propagate.

**Common rules:**

| Operation | Forward | Backward |
|-----------|---------|----------|
| x**2 | y = x² | dy/dx = 2x |
| x + y | z = x + y | dz/dx = 1, dz/dy = 1 |
| x * y | z = x * y | dz/dx = y, dz/dy = x |
| sin(x) | y = sin(x) | dy/dx = cos(x) |

**Partial derivatives** for multiple inputs:

```python
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)
z = x**2 + x*y + y**2

dz_dx = torch.autograd.grad(z, x, retain_graph=True)[0]
dz_dy = torch.autograd.grad(z, y)[0]
```

---

## retain_graph

By default, `backward()` frees the computation graph after execution. Use **retain_graph=True** when you need to call `backward()` multiple times on the same graph.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = y**2

z.backward(retain_graph=True)
grad_first = x.grad.clone()

x.grad.zero_()
z.backward()
grad_second = x.grad.clone()
```

---

## Gradient Tracking Enable/Disable

Gradient tracking can be toggled conditionally:

```python
def conditional_forward(x, track_gradients=True):
    if track_gradients:
        y = x**2 + 2*x + 1
    else:
        with torch.no_grad():
            y = x**2 + 2*x + 1
    return y

x = torch.tensor([1.0, 2.0], requires_grad=True)
y_with_grad = conditional_forward(x, True)
y_no_grad = conditional_forward(x, False)
```

---

## .grad Attribute

The **.grad** attribute stores the accumulated gradient for leaf tensors. Non-leaf tensors do not retain gradients by default; use **retain_grad()** to force retention.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2
y.retain_grad()
z = y.sum()
z.backward()

print(x.grad)
print(y.grad)
```

**Gradient accumulation:** Gradients accumulate across backward calls. Call `optimizer.zero_grad()` or `param.grad.zero_()` before each training step.

---

## .grad_fn Chain

Each non-leaf tensor has a **grad_fn** attribute pointing to the backward function that created it. The **next_functions** attribute links to parent nodes in the graph.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**3
z = torch.log(y)
w = z * 2

print(y.grad_fn)
print(z.grad_fn)
print(w.grad_fn)
print(w.grad_fn.next_functions)
```

**Detaching** breaks the gradient chain:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2
y_detached = y.detach()
z = y_detached * 3
z.sum().backward()
print(x.grad)
```

**Best practices:**
- Use `torch.no_grad()` for inference
- Clear gradients with `optimizer.zero_grad()` before each step
- Avoid in-place operations on tensors with `requires_grad=True`
- Use `retain_grad()` sparingly
- Use `detach()` when breaking gradient flow
