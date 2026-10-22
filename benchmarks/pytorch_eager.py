"""Per-operation cost of PyTorch eager operations (host overhead for small tensors, GPU time for large ones).

    LD_PRELOAD=build/lib/libcudart.so.13:build/lib/libcuda.so.1 python3 benchmarks/pytorch_eager.py
"""
import torch, time
x = torch.randn(1024, device="cuda")
big = torch.randn(1 << 24, device="cuda")
def timeit(fn, n):
    fn(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6
print(f"small add      {timeit(lambda: x.add_(1), 5000):8.1f} us/op")
print(f"small sum      {timeit(lambda: x.sum(), 5000):8.1f} us/op")
print(f"64 MB add      {timeit(lambda: big.add_(1), 50):8.1f} us/op")
print(f"64 MB sum      {timeit(lambda: big.sum(), 50):8.1f} us/op")
print(f"item() sync    {timeit(lambda: x[0].item(), 2000):8.1f} us/op")
