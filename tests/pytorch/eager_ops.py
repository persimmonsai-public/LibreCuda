"""PyTorch eager operations on the GPU, each compared with the CPU. Matrix multiplication needs cuBLAS, which doesn't
run on LibreCUDA (it is expected to fail); everything else should pass.

    LD_PRELOAD=build/lib/libcudart.so.13:build/lib/libcuda.so.1 python3 tests/pytorch/eager_ops.py [-v]

-v prints the traceback of failures. Without LD_PRELOAD the script runs on NVIDIA's CUDA (all cases pass there).
"""
import sys, time, traceback
import torch

dev = "cuda"
torch.manual_seed(0)
results = []


def case(name):
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
            torch.cuda.synchronize()
            results.append((name, "ok", time.time() - t0, ""))
            print(f"ok    {name} ({time.time() - t0:.2f}s)", flush=True)
        except Exception as e:
            results.append((name, "FAIL", time.time() - t0, str(e).splitlines()[0][:200]))
            print(f"FAIL  {name}: {str(e).splitlines()[0][:300]}", flush=True)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


def same(gpu, cpu, rtol=1e-4, atol=1e-5):
    g = gpu.cpu()
    if g.dtype.is_floating_point or g.dtype.is_complex:
        assert torch.allclose(g, cpu, rtol=rtol, atol=atol, equal_nan=True), f"max diff {(g - cpu).abs().max()}"
    else:
        assert torch.equal(g, cpu), "values differ"


A = torch.randn(1000, 37)
B = torch.randn(1000, 37)
I = torch.randint(-50, 50, (4096,))


@case("tensor creation and copies")
def _():
    same(torch.zeros(5, 7, device=dev), torch.zeros(5, 7))
    same(torch.ones(3, device=dev, dtype=torch.float64) * 2.5, torch.ones(3, dtype=torch.float64) * 2.5)
    same(torch.full((4, 4), 3, device=dev, dtype=torch.int32), torch.full((4, 4), 3, dtype=torch.int32))
    same(torch.arange(0, 100, 3, device=dev), torch.arange(0, 100, 3))
    same(torch.linspace(-1, 1, 101, device=dev), torch.linspace(-1, 1, 101))
    same(A.to(dev), A)
    x = A.to(dev)
    same(x.clone().t().contiguous(), A.t().contiguous())
    same(x.to(torch.float16).float(), A.half().float(), rtol=1e-3, atol=1e-3)
    same(x.to(torch.bfloat16).float(), A.bfloat16().float(), rtol=1e-2, atol=1e-2)
    pinned = torch.randn(1 << 16).pin_memory()
    same(pinned.to(dev, non_blocking=True), pinned)


@case("elementwise arithmetic and math")
def _():
    a, b = A.to(dev), B.to(dev)
    same(a + b, A + B)
    same(a * b - a / (b.abs() + 1), A * B - A / (B.abs() + 1))
    for f in [torch.exp, torch.log1p, torch.sin, torch.cos, torch.tanh, torch.sigmoid, torch.erf, torch.sqrt,
              torch.rsqrt, torch.floor, torch.round]:
        same(f(a.abs()), f(A.abs()), rtol=1e-4, atol=1e-5)
    same(torch.pow(a.abs(), 1.7), torch.pow(A.abs(), 1.7))
    same(torch.clamp(a, -0.5, 0.5), torch.clamp(A, -0.5, 0.5))
    same(torch.where(a > b, a, b), torch.where(A > B, A, B))
    same(torch.maximum(a, b), torch.maximum(A, B))
    i = I.to(dev)
    same(i * 3 + 7, I * 3 + 7)
    same(i // 7, I // 7)
    same(i % 7, I % 7)
    same(i & 0x0f, I & 0x0f)
    same(i.float().lerp(i.float() * 2, 0.25), I.float().lerp(I.float() * 2, 0.25))


@case("in-place and broadcasting")
def _():
    a = A.to(dev)
    c = A.clone()
    a.mul_(2).add_(torch.arange(37, device=dev, dtype=torch.float32))
    c.mul_(2).add_(torch.arange(37, dtype=torch.float32))
    same(a, c)
    same(a[:, :1] + a[:1, :], c[:, :1] + c[:1, :])


@case("reductions")
def _():
    a = A.to(dev)
    same(a.sum(), A.sum(), rtol=1e-4, atol=1e-3)
    same(a.sum(0), A.sum(0), rtol=1e-4, atol=1e-4)
    same(a.mean(1), A.mean(1))
    same(a.std(0), A.std(0), rtol=1e-4, atol=1e-4)
    same(a.var(), A.var(), rtol=1e-4, atol=1e-4)
    same(a.max(), A.max())
    same(a.min(1).values, A.min(1).values)
    same(a.argmax(0), A.argmax(0))
    same(a.norm(), A.norm(), rtol=1e-4, atol=1e-3)
    same(a.prod(1), A.prod(1), rtol=1e-3, atol=1e-6)
    same((a > 0).sum(), (A > 0).sum())
    same((a > 0).any(), (A > 0).any())
    same(a.logsumexp(1), A.logsumexp(1))
    same(torch.cumsum(a, 0), torch.cumsum(A, 0), rtol=1e-4, atol=1e-3)
    same(torch.cumprod(a[:20] * 0.1 + 1, 0), torch.cumprod(A[:20] * 0.1 + 1, 0), rtol=1e-4, atol=1e-4)
    big = torch.randn(1 << 22)
    same(big.to(dev).sum(), big.sum(), rtol=1e-3, atol=1e-2)


@case("indexing, gather and scatter")
def _():
    a = A.to(dev)
    idx = torch.randint(0, 1000, (300,))
    same(a[idx.to(dev)], A[idx])
    same(a[a > 0.5], A[A > 0.5])
    same(a[10:500:3, 5:30:2], A[10:500:3, 5:30:2])
    g = torch.randint(0, 37, (1000, 5))
    same(torch.gather(a, 1, g.to(dev)), torch.gather(A, 1, g))
    z = torch.zeros(10, 37)
    t = torch.randint(0, 10, (1000, 37))
    same(torch.zeros(10, 37, device=dev).scatter_add_(0, t.to(dev), a), z.scatter_add_(0, t, A), rtol=1e-4,
         atol=1e-4)
    c = A.clone()
    x = a.clone()
    x[idx.to(dev)] = 0
    c[idx] = 0
    same(x, c)
    same(torch.index_select(a, 1, torch.tensor([3, 1, 4], device=dev)), torch.index_select(A, 1, torch.tensor([3, 1, 4])))
    same(torch.nonzero(a > 1.5), torch.nonzero(A > 1.5))
    same(torch.masked_fill(a, a < 0, -1), torch.masked_fill(A, A < 0, -1))
    v = torch.zeros(1000)
    same(torch.zeros(1000, device=dev).index_put_((idx.to(dev),), torch.ones(300, device=dev), accumulate=True),
         v.index_put_((idx,), torch.ones(300), accumulate=True))


@case("sort, topk, unique")
def _():
    a = A.to(dev)
    same(torch.sort(a, 1).values, torch.sort(A, 1).values)
    same(torch.sort(a.flatten()).values, torch.sort(A.flatten()).values)
    same(torch.argsort(I.to(dev), stable=True), torch.argsort(I, stable=True))
    same(torch.topk(a.flatten(), 10).values, torch.topk(A.flatten(), 10).values)
    same(torch.unique(I.to(dev)), torch.unique(I))
    same(torch.kthvalue(a, 5, 1).values, torch.kthvalue(A, 5, 1).values)
    same(torch.searchsorted(torch.arange(100, device=dev), I.to(dev).clamp(0, 99)),
         torch.searchsorted(torch.arange(100), I.clamp(0, 99)))


@case("shape ops")
def _():
    a, b = A.to(dev), B.to(dev)
    same(torch.cat([a, b], 0), torch.cat([A, B], 0))
    same(torch.stack([a, b], 2), torch.stack([A, B], 2))
    same(a.reshape(37, 1000).permute(1, 0).contiguous(), A.reshape(37, 1000).permute(1, 0).contiguous())
    same(torch.flip(a, [0, 1]), torch.flip(A, [0, 1]))
    same(torch.roll(a, 5, 0), torch.roll(A, 5, 0))
    same(torch.repeat_interleave(I[:10].to(dev).abs(), 3), torch.repeat_interleave(I[:10].abs(), 3))
    same(torch.nn.functional.pad(a, (1, 2, 3, 4), value=1.5), torch.nn.functional.pad(A, (1, 2, 3, 4), value=1.5))
    same(a.split(100)[3], A.split(100)[3])
    same(torch.tril(a[:37]), torch.tril(A[:37]))


@case("random numbers")
def _():
    torch.cuda.manual_seed(1)
    u = torch.rand(1 << 20, device=dev)
    assert 0 <= u.min().item() and u.max().item() < 1 and abs(u.mean().item() - 0.5) < 0.01
    n = torch.randn(1 << 20, device=dev)
    assert abs(n.mean().item()) < 0.01 and abs(n.std().item() - 1) < 0.01
    torch.cuda.manual_seed(1)
    assert torch.equal(torch.rand(1 << 20, device=dev), u), "same seed, other numbers"
    p = torch.randperm(1000, device=dev)
    assert torch.equal(p.sort().values.cpu(), torch.arange(1000))
    m = torch.multinomial(torch.tensor([0.1, 0.0, 0.9], device=dev), 1000, replacement=True)
    assert (m != 1).all().item()
    d = torch.nn.functional.dropout(torch.ones(10000, device=dev), 0.5)
    assert 0.4 < (d == 0).float().mean().item() < 0.6


@case("neural network layers without cuBLAS / cuDNN")
def _():
    a = A.to(dev)
    for f in [torch.relu, torch.nn.functional.gelu, torch.nn.functional.silu, torch.nn.functional.softplus]:
        same(f(a), f(A), rtol=1e-4, atol=1e-5)
    same(torch.softmax(a, 1), torch.softmax(A, 1))
    same(torch.log_softmax(a, 0), torch.log_softmax(A, 0), rtol=1e-4, atol=1e-4)
    same(torch.nn.functional.layer_norm(a, (37,)), torch.nn.functional.layer_norm(A, (37,)), rtol=1e-4, atol=1e-4)
    t = torch.randint(0, 37, (1000,))
    same(torch.nn.functional.cross_entropy(a, t.to(dev)), torch.nn.functional.cross_entropy(A, t), rtol=1e-4,
         atol=1e-5)
    emb = torch.randn(50, 8)
    same(torch.nn.functional.embedding(t.to(dev), emb.to(dev)), torch.nn.functional.embedding(t, emb))
    img = torch.randn(2, 3, 16, 16)
    same(torch.nn.functional.max_pool2d(img.to(dev), 2), torch.nn.functional.max_pool2d(img, 2))
    same(torch.nn.functional.avg_pool2d(img.to(dev), 2), torch.nn.functional.avg_pool2d(img, 2))
    same(torch.nn.functional.interpolate(img.to(dev), scale_factor=2, mode="bilinear"),
         torch.nn.functional.interpolate(img, scale_factor=2, mode="bilinear"), rtol=1e-4, atol=1e-5)


@case("autograd")
def _():
    x = A.clone().requires_grad_()
    xg = A.to(dev).requires_grad_()
    w = torch.randn(37)
    wg = w.to(dev).requires_grad_()
    w.requires_grad_()
    loss = (torch.tanh(x * w).pow(2).sum(1).sqrt() + torch.softmax(x, 1)[:, 0]).mean()
    lossg = (torch.tanh(xg * wg).pow(2).sum(1).sqrt() + torch.softmax(xg, 1)[:, 0]).mean()
    loss.backward()
    lossg.backward()
    same(lossg.detach(), loss.detach())
    same(xg.grad, x.grad, rtol=1e-4, atol=1e-6)
    same(wg.grad, w.grad, rtol=1e-4, atol=1e-5)


@case("training loop of an elementwise model")
def _():
    def train(device):
        x = torch.linspace(-2, 2, 4096, device=device)
        y = 3 * x.pow(3) - x + 0.5
        params = torch.zeros(4, device=device, requires_grad=True)
        opt = torch.optim.Adam([params], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            pred = params[0] * x.pow(3) + params[1] * x.pow(2) + params[2] * x + params[3]
            loss = (pred - y).pow(2).mean()
            loss.backward()
            opt.step()
        return params.detach(), loss.detach()
    params, loss = train(dev)
    cpu_params, cpu_loss = train("cpu")
    same(params, cpu_params, rtol=1e-3, atol=1e-3)
    same(loss, cpu_loss, rtol=1e-3, atol=1e-4)


@case("streams and events")
def _():
    s = torch.cuda.Stream()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a = A.to(dev)
    with torch.cuda.stream(s):
        e0.record()
        b = (a * 2).sum(1)
        e1.record()
    torch.cuda.current_stream().wait_stream(s)
    same(b, (A * 2).sum(1), rtol=1e-4, atol=1e-4)
    e1.synchronize()
    assert e0.elapsed_time(e1) >= 0
    assert s.query()


@case("memory management")
def _():
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    xs = [torch.empty(1 << 20, device=dev) for _ in range(50)]
    assert torch.cuda.memory_allocated() - before == 50 * (1 << 22)
    del xs
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    assert 0 < free <= total
    assert torch.cuda.memory_allocated() == before


@case("complex numbers and jiterator ops")
def _():
    c = torch.randn(1000, dtype=torch.complex64)
    cg = c.to(dev)
    same(cg * cg.conj(), c * c.conj(), rtol=1e-4, atol=1e-5)
    same(torch.abs(cg), torch.abs(c))
    same(torch.exp(cg), torch.exp(c), rtol=1e-4, atol=1e-5)
    # torch.special functions that PyTorch compiles at run time with NVRTC (jiterator)
    x = torch.rand(1000) * 5
    same(torch.special.i0e(x.to(dev)), torch.special.i0e(x), rtol=1e-4, atol=1e-6)
    same(torch.special.bessel_j0(x.to(dev)), torch.special.bessel_j0(x), rtol=1e-3, atol=1e-5)


@case("matmul (cuBLAS, expected to fail)")
def _():
    a = A.to(dev)
    same(a.t() @ a, A.t() @ A, rtol=1e-3, atol=1e-2)


fails = [r for r in results if r[1] != "ok"]
print(f"\n{len(results) - len(fails)} of {len(results)} passed")
for name, _, _, msg in fails:
    print(f"  {name}: {msg}")
sys.exit(0 if all(name.endswith("expected to fail)") for name, _, _, _ in fails) else 1)
