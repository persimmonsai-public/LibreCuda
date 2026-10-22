"""torch.fft on the GPU (LibreCUDA's cuFFT), each compared with the CPU:

    L=build/lib
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcufft.so.12 \\
        python3 tests/pytorch/fft_ops.py [-v]

Without LD_PRELOAD the script runs on NVIDIA's CUDA.
"""
import sys
import time
import traceback

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
            results.append((name, None))
            print(f"ok    {name} ({time.time() - t0:.2f}s)", flush=True)
        except Exception as e:
            results.append((name, str(e).splitlines()[0][:300]))
            print(f"FAIL  {name}: {str(e).splitlines()[0][:300]}", flush=True)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


def close(a, b, tol, what=""):
    a, b = a.detach().cpu().to(torch.complex128 if a.is_complex() else torch.float64), b.detach().cpu()
    b = b.to(a.dtype)
    err = (a - b).abs().max().item() if a.numel() else 0
    scale = max(b.abs().max().item() if b.numel() else 1, 1)
    assert err <= tol * scale, f"{what}: max diff {err:.3g} (scale {scale:.3g})"


SIZES = [1, 2, 8, 16, 1024, 6, 12, 15, 60, 360, 1000, 7, 49, 13, 11 * 13, 127, 997, 2 * 127, 4096 * 3, 65536]


@case("1D complex and real transforms of many sizes (mixed radix, Bluestein)")
def _():
    for n in SIZES:
        for dtype, tol in [(torch.complex64, 2e-5), (torch.complex128, 1e-12)]:
            x = torch.randn(3, n, dtype=dtype)
            close(torch.fft.fft(x.to(dev)), torch.fft.fft(x), tol * max(1, n) ** 0.5, f"fft {n}")
            close(torch.fft.ifft(x.to(dev)), torch.fft.ifft(x), tol * max(1, n) ** 0.5, f"ifft {n}")
            r = x.real.contiguous()
            close(torch.fft.rfft(r.to(dev)), torch.fft.rfft(r), tol * max(1, n) ** 0.5, f"rfft {n}")
            spec = torch.fft.rfft(r)
            close(torch.fft.irfft(spec.to(dev), n), torch.fft.irfft(spec, n), tol * max(1, n) ** 0.5, f"irfft {n}")


@case("2D and N-D transforms, norms, dimensions")
def _():
    x = torch.randn(2, 12, 10, 7, dtype=torch.complex64)
    close(torch.fft.fft2(x.to(dev)), torch.fft.fft2(x), 1e-4)
    close(torch.fft.fftn(x.to(dev), dim=(1, 2, 3)), torch.fft.fftn(x, dim=(1, 2, 3)), 1e-4)
    close(torch.fft.ifftn(x.to(dev), norm="ortho"), torch.fft.ifftn(x, norm="ortho"), 1e-4)
    close(torch.fft.fft(x.to(dev), dim=1, norm="forward"), torch.fft.fft(x, dim=1, norm="forward"), 1e-4)
    r = torch.randn(3, 16, 9, 6)
    close(torch.fft.rfftn(r.to(dev), dim=(1, 2, 3)), torch.fft.rfftn(r, dim=(1, 2, 3)), 1e-4)
    s = torch.fft.rfft2(r)
    close(torch.fft.irfft2(s.to(dev), r.shape[-2:]), torch.fft.irfft2(s, r.shape[-2:]), 1e-4)
    close(torch.fft.hfft(x[0, 0].to(dev), 13), torch.fft.hfft(x[0, 0], 13), 1e-4)
    close(torch.fft.fftshift(torch.fft.fft2(r.to(dev))), torch.fft.fftshift(torch.fft.fft2(r)), 1e-4)


@case("non-contiguous inputs, padding and trimming")
def _():
    x = torch.randn(20, 33, dtype=torch.complex64)
    xs = x[::2, 1::3]
    close(torch.fft.fft(xs.to(dev)), torch.fft.fft(xs), 1e-5)
    close(torch.fft.fft(x.t().to(dev), dim=0), torch.fft.fft(x.t(), dim=0), 1e-5)
    close(torch.fft.fft(x.to(dev), n=50), torch.fft.fft(x, n=50), 1e-5)
    close(torch.fft.rfft(x.real.to(dev), n=21), torch.fft.rfft(x.real, n=21), 1e-5)


@case("half precision (powers of two)")
def _():
    x = torch.randn(4, 256)
    close(torch.fft.rfft(x.to(dev).half()), torch.fft.rfft(x.half().float()), 2e-3)
    c = torch.randn(4, 64, dtype=torch.complex64)
    close(torch.fft.fft(c.to(dev).to(torch.complex32)).to(torch.complex64),
          torch.fft.fft(c.to(torch.complex32).to(torch.complex64)), 2e-3)


@case("STFT, inverse STFT, gradients")
def _():
    signal = torch.randn(2, 4000)
    window = torch.hann_window(400)
    spec_c = torch.stft(signal, 400, 160, window=window, return_complex=True)
    spec_g = torch.stft(signal.to(dev), 400, 160, window=window.to(dev), return_complex=True)
    close(spec_g, spec_c, 1e-4)
    close(torch.istft(spec_g, 400, 160, window=window.to(dev), length=4000),
          torch.istft(spec_c, 400, 160, window=window, length=4000), 1e-4)
    x = torch.randn(5, 30, requires_grad=True)
    xg = x.detach().to(dev).requires_grad_()
    torch.fft.rfft(x).abs().pow(2).sum().backward()
    torch.fft.rfft(xg).abs().pow(2).sum().backward()
    close(xg.grad, x.grad, 1e-4)


fails = [(n, m) for n, m in results if m is not None]
print(f"\n{len(results) - len(fails)} of {len(results)} passed")
for name, msg in fails:
    print(f"  {name}: {msg}")
sys.exit(1 if fails else 0)
