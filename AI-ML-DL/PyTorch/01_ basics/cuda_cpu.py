import torch
import time

N = 1000

def cpu_multiplication():
  device = torch.device("cpu")
  print(device)
  A = torch.randn(N, N, device=device)
  B = torch.randn(N, N, device=device)
  start = time.time()
  C = torch.matmul(A, B)
  end = time.time()
  print(f"CPU Time: { end - start} seconds")

def gpu_multiplication():
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  print(device)
  # device = torch.device("cuda")
  A = torch.randn(N, N, device= device)
  B = torch.randn(N, N, device= device)
  start = time.time()
  C = torch.matmul(A, B)
  if device == "cuda":
    torch.cuda.synchronize()
  end = time.time()
  print(f"GPU Time: {end - start} seconds")


cpu_multiplication()
gpu_multiplication()
