"""torch.compile (TorchInductor generating Triton kernels) on the GPU, each compared with eager mode:

    L=build/lib
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13 \\
        python3 tests/pytorch/compile_ops.py [-v]

Caches are disabled so every kernel is compiled (ptxas) and loaded (cuModuleLoadData) in this run. -v prints the
traceback of failures. Without LD_PRELOAD the script runs on NVIDIA's CUDA.
"""
import importlib
import os
import sys
import time
import traceback

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
import torch
import torch._dynamo
import torch._inductor.config

dev = "cuda"
torch.manual_seed(0)
results = []


def case(name):
    def wrap(fn):
        torch._dynamo.reset()
        t0 = time.time()
        try:
            fn()
            torch.cuda.synchronize()
            results.append((name, None))
            print(f"ok    {name} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:
            results.append((name, str(e).splitlines()[0][:300]))
            print(f"FAIL  {name}: {str(e).splitlines()[0][:300]}", flush=True)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


def same(a, b, rtol=1e-4, atol=1e-4):
    if isinstance(a, (tuple, list)):
        for x, y in zip(a, b):
            same(x, y, rtol, atol)
        return
    assert torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol), f"max diff {(a.float() - b.float()).abs().max()}"


@case("pointwise fusion")
def _():
    def f(x, y):
        return torch.sigmoid(x) * torch.tanh(y) + torch.exp(-x * x) / (1 + y.abs())
    x, y = torch.randn(4096, 513, device=dev), torch.randn(4096, 513, device=dev)
    same(torch.compile(f)(x, y), f(x, y))


@case("reductions and normalization")
def _():
    def f(x, w, b):
        h = torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b)
        return torch.softmax(h, -1).amax(-1) + h.var(-1) + x.sum(0).mean()
    x, w, b = torch.randn(512, 1000, device=dev), torch.randn(1000, device=dev), torch.randn(1000, device=dev)
    same(torch.compile(f)(x, w, b), f(x, w, b))


@case("dynamic shapes")
def _():
    def f(x):
        return (x.cumsum(0) * torch.arange(x.shape[0], device=x.device)[:, None]).relu().sum(1)
    cf = torch.compile(f, dynamic=True)
    for n in (17, 100, 1023):
        x = torch.randn(n, 33, device=dev)
        same(cf(x), f(x), rtol=1e-4, atol=1e-3)


@case("indexing and scatter")
def _():
    def f(x, idx, src):
        y = x.index_add(0, idx, src)
        return y[idx] * 2 + torch.gather(y, 1, (idx[:, None] % y.shape[1]).expand(-1, 4))[:, :1]
    x = torch.randn(1000, 64, device=dev)
    idx = torch.randint(0, 1000, (300,), device=dev)
    src = torch.randn(300, 64, device=dev)
    same(torch.compile(f)(x, idx, src), f(x, idx, src))


@case("compiled backward")
def _():
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.LayerNorm(128),
                                torch.nn.Linear(128, 10)).to(dev)
    x = torch.randn(256, 64, device=dev)
    y = torch.randint(0, 10, (256,), device=dev)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    cmodel = torch.compile(model)
    closs = torch.nn.functional.cross_entropy(cmodel(x), y)
    cgrads = torch.autograd.grad(closs, list(model.parameters()))
    same(closs, loss)
    same(cgrads, grads, rtol=1e-3, atol=1e-5)


@case("max-autotune: Triton matrix multiplication templates")
def _():
    def f(a, b, bias):
        return torch.relu(a @ b + bias)
    a, b = torch.randn(512, 384, device=dev, dtype=torch.float16), torch.randn(384, 256, device=dev,
                                                                             dtype=torch.float16)
    bias = torch.randn(256, device=dev, dtype=torch.float16)
    # Inductor offers Triton GEMM templates only on GPUs with 68 SMs or more: allow them on smaller ones
    importlib.import_module("torch._inductor.utils").is_big_gpu = lambda *args, **kwargs: True
    with torch._inductor.config.patch(max_autotune_gemm_backends="TRITON"):
        cf = torch.compile(f, mode="max-autotune-no-cudagraphs")
        same(cf(a, b, bias), f(a, b, bias), rtol=1e-2, atol=1e-2)


@case("reduce-overhead: CUDA graphs")
def _():
    model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32)).to(dev)
    cmodel = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        for i in range(5):
            x = torch.randn(16, 32, device=dev)
            same(cmodel(x), model(x))


@case("transformer training step")
def _():
    torch.manual_seed(1)
    layer = torch.nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True).to(dev)
    clayer = torch.compile(layer)
    x = torch.randn(8, 32, 128, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
        cout = clayer(x)
    same(cout, out, rtol=5e-2, atol=5e-2)
    cout.float().pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


fails = [(n, m) for n, m in results if m is not None]
print(f"\n{len(results) - len(fails)} of {len(results)} passed")
for name, msg in fails:
    print(f"  {name}: {msg}")
sys.exit(1 if fails else 0)
