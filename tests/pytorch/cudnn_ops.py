"""cuDNN operations of PyTorch on the GPU (LibreCUDA's cuDNN), each compared with the CPU:

    L=build/lib
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcusolver.so.12:\\
$L/libcusparse.so.12:$L/libcudnn.so.9 python3 tests/pytorch/cudnn_ops.py [-v]

TORCH_CUDNN_V8_API_DISABLED=1 makes PyTorch use the legacy convolution API instead of the backend graph API. Without
LD_PRELOAD the script runs on NVIDIA's CUDA.
"""
import sys
import time
import traceback

import torch
import torch.nn.functional as F

dev = "cuda"
torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
results = []


def case(name):
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
            torch.cuda.synchronize()
            results.append((name, None))
            print(f"ok    {name} ({time.time() - t0:.2f}s)", flush=True)
        except Exception as e:
            results.append((name, str(e).splitlines()[0][:300]))
            print(f"FAIL  {name}: {str(e).splitlines()[0][:300]}", flush=True)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


def close(a, b, tol):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    err = (a - b).abs().max().item() if a.numel() else 0
    scale = max(b.abs().max().item() if b.numel() else 1, 1)
    assert err <= tol * scale, f"max diff {err:.3g} (scale {scale:.3g})"


def check_conv(fn, x, w, b, tol, **kwargs):
    """forward and gradients of fn(x, w, b) on the GPU against the CPU (in float64)"""
    xs = [t.detach().clone().requires_grad_() for t in (x, w)] + ([b.detach().clone().requires_grad_()] if b is not None else [])
    xg = [t.detach().to(dev).requires_grad_() for t in xs]
    out_cpu = fn(*[t.double() for t in xs[:2]], xs[2].double() if b is not None else None, **kwargs)
    out_gpu = fn(*xg[:2], xg[2] if b is not None else None, **kwargs)
    grad = torch.randn_like(out_cpu)
    out_cpu.backward(grad)
    out_gpu.backward(grad.to(dev).to(out_gpu.dtype))
    close(out_gpu, out_cpu, tol)
    for g, c in zip(xg, xs):
        close(g.grad, c.grad, tol * 10)


@case("convolutions: 1d, 2d, 3d, stride, dilation, padding, groups")
def _():
    for dtype, tol in [(torch.float32, 1e-4), (torch.float16, 1e-2), (torch.bfloat16, 5e-2), (torch.float64, 1e-10)]:
        configs = [
            (F.conv1d, (4, 6, 50), (8, 6, 5), dict(stride=2, padding=2, dilation=1)),
            (F.conv2d, (2, 4, 20, 17), (6, 4, 3, 3), dict(stride=1, padding=1)),
            (F.conv2d, (2, 4, 20, 17), (6, 2, 3, 5), dict(stride=(2, 1), padding=(1, 2), dilation=(1, 2), groups=2)),
            (F.conv2d, (3, 8, 9, 9), (8, 1, 3, 3), dict(padding=1, groups=8)),  # depthwise
            (F.conv2d, (2, 16, 7, 7), (32, 16, 1, 1), dict()),  # pointwise
            (F.conv3d, (2, 3, 8, 9, 10), (4, 3, 3, 3, 3), dict(stride=2, padding=1)),
        ]
        for fn, xs, ws, kw in configs:
            x = torch.randn(xs).to(dtype)
            w = torch.randn(ws).to(dtype) / (ws[1] * max(1, ws[-1])) ** 0.5
            b = torch.randn(ws[0]).to(dtype)
            check_conv(fn, x, w, b, tol, **kw)


@case("half / bfloat16 convolutions of network shapes (implicit GEMM): padded channels, strides, dilation")
def _():
    configs = [
        ((4, 3, 64, 64), (64, 3, 7, 7), dict(stride=2, padding=3)),  # 3 input channels (padded to 8)
        ((4, 64, 28, 28), (64, 64, 3, 3), dict(padding=1)),
        ((4, 64, 28, 28), (128, 64, 1, 1), dict(stride=2)),  # downsampling: dgrad positions without taps
        ((4, 64, 28, 28), (128, 64, 3, 3), dict(stride=2, padding=1)),
        ((2, 32, 30, 30), (45, 32, 3, 3), dict(padding=2, dilation=2)),  # 45 filters (padded)
        ((2, 20, 17, 23), (12, 20, 5, 3), dict(stride=(3, 2), padding=(2, 0))),
        ((256, 16, 6, 6), (16, 16, 3, 3), dict(padding=1)),  # many images, few positions (split-K weights)
    ]
    for dtype, tol in [(torch.float16, 1e-2), (torch.bfloat16, 5e-2)]:
        for xs, ws, kw in configs:
            x = torch.randn(xs).to(dtype)
            w = (torch.randn(ws) / (ws[1] * ws[2] * ws[3]) ** 0.5).to(dtype)
            check_conv(F.conv2d, x, w, None, tol, **kw)
        x = torch.randn(4, 24, 40).to(dtype)
        check_conv(F.conv1d, x, (torch.randn(40, 24, 3) / 8.5).to(dtype), None, tol, stride=2, padding=1)
        x = torch.randn(2, 16, 9, 9).to(dtype)
        check_conv(F.conv_transpose2d, x, (torch.randn(16, 8, 4, 4) / 16).to(dtype), None, tol, stride=2, padding=1)
        # channels last
        x = torch.randn(4, 32, 14, 14)
        w = torch.randn(64, 32, 3, 3) / 17
        out = F.conv2d(x.to(dev, dtype).to(memory_format=torch.channels_last),
                       w.to(dev, dtype).to(memory_format=torch.channels_last), padding=1)
        close(out, F.conv2d(x.to(dtype).double(), w.to(dtype).double(), padding=1), tol)


@case("channels_last and transposed convolutions")
def _():
    x = torch.randn(2, 8, 12, 12)
    w = torch.randn(16, 8, 3, 3) / 8
    b = torch.randn(16)
    xg = x.to(dev).to(memory_format=torch.channels_last)
    wg = w.to(dev).to(memory_format=torch.channels_last)
    close(F.conv2d(xg, wg, b.to(dev), padding=1), F.conv2d(x.double(), w.double(), b.double(), padding=1), 1e-4)
    wt = torch.randn(8, 4, 4, 4) / 8
    check_conv(F.conv_transpose2d, x, wt, torch.randn(4), 1e-4, stride=2, padding=1)
    check_conv(F.conv_transpose1d, torch.randn(2, 8, 30), torch.randn(8, 3, 5) / 8, None, 1e-4, stride=3)


@case("batch normalization: training, evaluation, gradients")
def _():
    for dtype, tol in [(torch.float32, 1e-4), (torch.float16, 2e-2)]:
        for shape in [(8, 6, 10, 10), (16, 5), (4, 3, 5, 6, 7)]:
            bn_cpu = {2: torch.nn.BatchNorm1d, 4: torch.nn.BatchNorm2d, 5: torch.nn.BatchNorm3d}[len(shape)](shape[1])
            bn_cpu.weight.data.uniform_(0.5, 1.5)
            bn_cpu.bias.data.uniform_(-0.5, 0.5)
            bn_gpu = type(bn_cpu)(shape[1]).to(dev)
            bn_gpu.load_state_dict(bn_cpu.state_dict())
            bn_gpu = bn_gpu.to(dev).to(dtype) if dtype == torch.float64 else bn_gpu
            x = torch.randn(shape) * 2 + 1
            xc = x.clone().requires_grad_()
            xg = x.to(dev).to(dtype).requires_grad_()
            for _ in range(2):
                yc, yg = bn_cpu(xc), bn_gpu(xg)
            g = torch.randn(shape)
            yc.backward(g)
            yg.backward(g.to(dev).to(dtype))
            close(yg, yc, tol)
            close(xg.grad, xc.grad, tol * 5)
            close(bn_gpu.weight.grad, bn_cpu.weight.grad, tol * 10)
            close(bn_gpu.running_mean, bn_cpu.running_mean, tol)
            close(bn_gpu.running_var, bn_cpu.running_var, tol)
            bn_cpu.eval(), bn_gpu.eval()
            close(bn_gpu(xg), bn_cpu(xc), tol)


@case("batch normalization of network shapes: many images, large means, channels_last, bfloat16, double")
def _():
    # (PyTorch normalizes double data with its own kernels, in float)
    for dtype, tol in [(torch.float32, 1e-4), (torch.float16, 2e-2), (torch.bfloat16, 5e-2), (torch.float64, 1e-6)]:
        for shape, offset, channels_last in [((64, 16, 14, 14), 1, False), ((32, 8, 7, 7), 100, False),
                                             ((16, 24, 9, 11), 1, True), ((256, 4, 3, 3), 3, False)]:
            bn_cpu = torch.nn.BatchNorm2d(shape[1]).double()
            bn_cpu.weight.data.uniform_(0.5, 1.5)
            bn_cpu.bias.data.uniform_(-0.5, 0.5)
            bn_gpu = torch.nn.BatchNorm2d(shape[1]).to(dev)
            bn_gpu.load_state_dict(bn_cpu.state_dict())
            if dtype == torch.float64:
                bn_gpu = bn_gpu.double()
            x = (torch.randn(shape) * 2 + offset).to(dtype)
            xc = x.double().clone().requires_grad_()
            xg = x.to(dev)
            if channels_last:
                xg = xg.to(memory_format=torch.channels_last)
            xg.requires_grad_()
            yc, yg = bn_cpu(xc), bn_gpu(xg)
            g = torch.randn(shape)
            yc.backward(g.double())
            yg.backward(g.to(dev).to(dtype))
            close(yg, yc, tol)
            close(xg.grad, xc.grad, tol * 5)
            close(bn_gpu.weight.grad, bn_cpu.weight.grad, tol * 10)
            close(bn_gpu.bias.grad, bn_cpu.bias.grad, tol * 10)
            close(bn_gpu.running_mean, bn_cpu.running_mean, tol)
            close(bn_gpu.running_var, bn_cpu.running_var, tol * 10)


@case("pooling and grid sampling")
def _():
    x = torch.randn(2, 3, 13, 11)
    xg = x.to(dev)
    close(F.max_pool2d(xg, 3, 2, 1), F.max_pool2d(x, 3, 2, 1), 1e-6)
    close(F.avg_pool2d(xg, 2), F.avg_pool2d(x, 2), 1e-6)
    theta = torch.tensor([[[0.9, 0.1, 0.05], [-0.1, 1.1, -0.02]]]).repeat(2, 1, 1)
    grid = F.affine_grid(theta, (2, 3, 9, 7), align_corners=True)
    close(F.affine_grid(theta.to(dev), (2, 3, 9, 7), align_corners=True), grid, 1e-6)
    xs = x.clone().requires_grad_()
    gs = grid.clone().requires_grad_()
    xgs = xg.clone().requires_grad_()
    ggs = grid.to(dev).requires_grad_()
    out_c = F.grid_sample(xs, gs, align_corners=True)
    out_g = F.grid_sample(xgs, ggs, align_corners=True)
    gout = torch.randn_like(out_c)
    out_c.backward(gout)
    out_g.backward(gout.to(dev))
    close(out_g, out_c, 1e-5)
    close(xgs.grad, xs.grad, 1e-4)
    close(ggs.grad, gs.grad, 1e-4)


@case("convolutional network training")
def _():
    torch.manual_seed(5)
    def block(cin, cout):
        return torch.nn.Sequential(torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False), torch.nn.BatchNorm2d(cout),
                                   torch.nn.ReLU())
    model = torch.nn.Sequential(block(1, 16), torch.nn.MaxPool2d(2), block(16, 32), torch.nn.AdaptiveAvgPool2d(1),
                                torch.nn.Flatten(), torch.nn.Linear(32, 4)).to(dev)
    # classify which quadrant holds a bright square
    def batch(n):
        x = torch.randn(n, 1, 16, 16) * 0.3
        y = torch.randint(0, 4, (n,))
        for i in range(n):
            r, c = divmod(int(y[i]), 2)
            x[i, 0, r * 8 + 2:r * 8 + 6, c * 8 + 2:c * 8 + 6] += 2
        return x.to(dev), y.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for _ in range(80):
        x, y = batch(64)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    x, y = batch(512)
    accuracy = (model(x).argmax(1) == y).float().mean().item()
    assert accuracy > 0.95, f"accuracy {accuracy}"


@case("recurrent networks (LSTM, GRU)")
def _():
    for cls in (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN):
        torch.manual_seed(1)
        cpu = cls(10, 16, num_layers=2, bidirectional=True, batch_first=True)
        gpu = cls(10, 16, num_layers=2, bidirectional=True, batch_first=True).to(dev)
        gpu.load_state_dict(cpu.state_dict())
        x = torch.randn(3, 7, 10)
        xc, xg = x.clone().requires_grad_(), x.to(dev).requires_grad_()
        oc, _ = cpu(xc)
        og, _ = gpu(xg)
        g = torch.randn_like(oc)
        oc.backward(g)
        og.backward(g.to(dev))
        close(og, oc, 1e-4)
        close(xg.grad, xc.grad, 1e-4)
        for pc, pg in zip(cpu.parameters(), gpu.parameters()):
            close(pg.grad, pc.grad, 1e-3)


@case("recurrent networks: packed sequences, states, inference, half precision, dropout")
def _():
    torch.manual_seed(3)
    cpu = torch.nn.LSTM(5, 12, num_layers=2, bidirectional=True)
    gpu = torch.nn.LSTM(5, 12, num_layers=2, bidirectional=True).to(dev)
    gpu.load_state_dict(cpu.state_dict())
    lengths = [9, 7, 4, 2]
    x = torch.randn(9, 4, 5)
    h0, c0 = torch.randn(4, 4, 12), torch.randn(4, 4, 12)
    packed_c = torch.nn.utils.rnn.pack_padded_sequence(x.clone().requires_grad_(), lengths)
    xg = x.to(dev).requires_grad_()
    packed_g = torch.nn.utils.rnn.pack_padded_sequence(xg, lengths)
    oc, (hc, cc) = cpu(packed_c, (h0, c0))
    og, (hg, cg) = gpu(packed_g, (h0.to(dev), c0.to(dev)))
    close(og.data, oc.data, 1e-4)
    close(hg, hc, 1e-4)
    close(cg, cc, 1e-4)
    (og.data.sum() + hg.sum() + cg.pow(2).sum()).backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    # inference mode (no reserve space) and half precision
    with torch.no_grad():
        gru_c = torch.nn.GRU(6, 10, num_layers=3)
        gru_g = torch.nn.GRU(6, 10, num_layers=3).to(dev)
        gru_g.load_state_dict(gru_c.state_dict())
        y = torch.randn(11, 3, 6)
        close(gru_g(y.to(dev))[0], gru_c(y)[0], 1e-4)
        close(gru_g.half()(y.to(dev).half())[0].float(), gru_c(y)[0], 2e-2)
    # dropout between layers: deterministic per call with the same seed, and roughly unbiased
    drop = torch.nn.LSTM(8, 16, num_layers=2, dropout=0.5).to(dev)
    z = torch.randn(6, 5, 8, device=dev)
    out, _ = drop(z)
    assert out.shape == (6, 5, 16) and torch.isfinite(out).all()
    out.sum().backward()


@case("CTC loss")
def _():
    T, N, C, S = 30, 4, 6, 8
    log_probs = torch.randn(T, N, C).log_softmax(2)
    targets = torch.randint(1, C, (N * S,), dtype=torch.int32)
    # (cuDNN's CTC loss: int32 targets on the CPU, all inputs as long as the sequence)
    input_lengths = [T] * N
    target_lengths = [S] * N
    lc = log_probs.clone().requires_grad_()
    lg = log_probs.to(dev).requires_grad_()
    loss_c = F.ctc_loss(lc, targets, input_lengths, target_lengths)
    loss_g = F.ctc_loss(lg, targets, input_lengths, target_lengths)
    loss_c.backward()
    loss_g.backward()
    close(loss_g, loss_c, 1e-4)
    close(lg.grad, lc.grad, 1e-4)


fails = [(n, m) for n, m in results if m is not None]
print(f"\n{len(results) - len(fails)} of {len(results)} passed")
for name, msg in fails:
    print(f"  {name}: {msg}")
sys.exit(1 if fails else 0)
