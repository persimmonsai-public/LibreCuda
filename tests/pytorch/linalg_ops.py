"""torch.linalg on the GPU (cuSOLVER and cuBLAS), each compared with the CPU:

    L=build/lib
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcusolver.so.12:\\
$L/libcusparse.so.12 python3 tests/pytorch/linalg_ops.py [-v]

Decompositions that aren't unique (eigenvectors, singular vectors, QR signs) are checked by their properties. Without
LD_PRELOAD the script runs on NVIDIA's CUDA.
"""
import sys
import time
import traceback

import torch

dev = "cuda"
torch.manual_seed(0)
results = []
EXPECTED_FAILURES = set()


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
    a, b = a.cpu(), b.cpu()
    err = (a - b).abs().max().item() if a.numel() else 0
    scale = max(b.abs().max().item() if b.numel() else 1, 1)
    assert err <= tol * scale, f"max diff {err:.3g} (scale {scale:.3g})"


DTYPES = [(torch.float32, 1e-3), (torch.float64, 1e-9), (torch.complex64, 1e-3), (torch.complex128, 1e-9)]


def matrices(shape, dtype):
    # (scaled so that determinants of large matrices stay finite)
    return torch.randn(*shape, dtype=dtype) / shape[-1] ** 0.5


def spd(shape, dtype):
    a = torch.randn(*shape, dtype=dtype)
    return a @ a.mH + shape[-1] * torch.eye(shape[-1], dtype=dtype)


@case("solve, inverse, determinant")
def _():
    for dtype, tol in DTYPES:
        for shape in [(5, 5), (4, 30, 30), (300, 300)]:
            a, b = matrices(shape, dtype), matrices(shape[:-1] + (3,), dtype)
            close(torch.linalg.solve(a.to(dev), b.to(dev)), torch.linalg.solve(a, b), tol * 10)
            close(torch.linalg.inv(a.to(dev)), torch.linalg.inv(a), tol * 10)
            close(torch.linalg.det(a.to(dev)), torch.linalg.det(a), tol * 10)
            sign, logdet = torch.linalg.slogdet(a.to(dev))
            sign_c, logdet_c = torch.linalg.slogdet(a)
            close(logdet, logdet_c, tol * 10)
            close(sign, sign_c, tol * 10)


@case("LU factorization and solve")
def _():
    for dtype, tol in DTYPES:
        for shape in [(6, 6), (3, 40, 40), (500, 500)]:
            a, b = matrices(shape, dtype), matrices(shape[:-1] + (2,), dtype)
            lu, piv = torch.linalg.lu_factor(a.to(dev))
            close(torch.linalg.lu_solve(lu, piv, b.to(dev)), torch.linalg.solve(a, b), tol * 10)
            p, l, u = torch.linalg.lu(a.to(dev))
            close(p @ l @ u, a, tol * 10)


@case("Cholesky")
def _():
    for dtype, tol in DTYPES:
        for shape in [(7, 7), (5, 20, 20), (400, 400)]:
            a, b = spd(shape, dtype), matrices(shape[:-1] + (2,), dtype)
            l = torch.linalg.cholesky(a.to(dev))
            close(l, torch.linalg.cholesky(a), tol)
            close(torch.cholesky_solve(b.to(dev), l), torch.cholesky_solve(b, l.cpu()), tol * 10)
            close(torch.linalg.cholesky(a.to(dev), upper=True), torch.linalg.cholesky(a, upper=True), tol)
            _, info = torch.linalg.cholesky_ex(-a.to(dev))
            assert (info.cpu() == 1).all(), f"info of a negative definite matrix: {info}"


@case("QR, householder product, least squares")
def _():
    for dtype, tol in DTYPES:
        for shape in [(8, 5), (3, 30, 20), (300, 200)]:
            a = matrices(shape, dtype)
            for mode in ("reduced", "complete"):
                q, r = torch.linalg.qr(a.to(dev), mode=mode)
                close(q @ r, a, tol * 10)
                eye = torch.eye(q.shape[-1], dtype=dtype)
                close(q.mH @ q, eye.expand(q.shape[:-2] + eye.shape), tol * 10)
                assert (r.cpu().tril(-1) == 0).all(), "R not upper triangular"
            geqrf, tau = torch.geqrf(a.to(dev))
            close(torch.linalg.householder_product(geqrf, tau), torch.linalg.qr(a.to(dev)).Q, tol * 10)
            b = matrices(shape[:-1] + (2,), dtype)
            close(torch.linalg.lstsq(a.to(dev), b.to(dev)).solution, torch.linalg.lstsq(a, b).solution, tol * 100)


@case("triangular solve, matrix exponential")
def _():
    for dtype, tol in DTYPES:
        a = torch.randn(40, 40, dtype=dtype).triu() + 5 * torch.eye(40, dtype=dtype)
        b = torch.randn(40, 7, dtype=dtype)
        for upper, left in [(True, True), (True, False)]:
            bb = b if left else b.mT
            close(torch.linalg.solve_triangular(a.to(dev), bb.to(dev), upper=upper, left=left),
                  torch.linalg.solve_triangular(a, bb, upper=upper, left=left), tol * 10)
        m = torch.randn(3, 20, 20, dtype=dtype) / 5
        close(torch.linalg.matrix_exp(m.to(dev)), torch.linalg.matrix_exp(m), tol * 10)


@case("symmetric eigen decomposition")
def _():
    for dtype, tol in DTYPES:
        for shape in [(6, 6), (8, 16, 16), (200, 200)]:
            a = matrices(shape, dtype)
            h = a + a.mH
            w, v = torch.linalg.eigh(h.to(dev))
            close(w, torch.linalg.eigvalsh(h), tol * 10)
            close(h.to(dev) @ v, v * w.to(v.dtype).unsqueeze(-2), tol * 10)
            close(torch.linalg.eigvalsh(h.to(dev), UPLO="U"), torch.linalg.eigvalsh(h, UPLO="U"), tol * 10)


@case("singular value decomposition, pseudo inverse, rank, norms")
def _():
    for dtype, tol in DTYPES:
        for shape in [(7, 5), (5, 7), (4, 12, 9), (300, 150)]:
            a = matrices(shape, dtype)
            for full in (False, True):
                u, s, vh = torch.linalg.svd(a.to(dev), full_matrices=full)
                k = min(shape[-2:])
                close(s, torch.linalg.svdvals(a), tol * 10)
                rebuilt = (u[..., :, :k] * s.to(u.dtype).unsqueeze(-2)) @ vh[..., :k, :]
                close(rebuilt, a, tol * 10)
                eye = torch.eye(u.shape[-1], dtype=dtype)
                close(u.mH @ u, eye.expand(u.shape[:-2] + eye.shape), tol * 10)
            close(torch.linalg.pinv(a.to(dev)), torch.linalg.pinv(a), tol * 100)
            close(torch.linalg.matrix_norm(a.to(dev), "nuc"), torch.linalg.matrix_norm(a, "nuc"), tol * 10)
            close(torch.linalg.matrix_norm(a.to(dev), 2), torch.linalg.matrix_norm(a, 2), tol * 10)
            assert torch.equal(torch.linalg.matrix_rank(a.to(dev)).cpu(), torch.linalg.matrix_rank(a))
        # a rank deficient matrix
        r = torch.randn(30, 4, dtype=dtype) @ torch.randn(4, 20, dtype=dtype)
        assert torch.linalg.matrix_rank(r.to(dev)).item() == 4


@case("autograd through decompositions")
def _():
    a = torch.randn(10, 10, dtype=torch.float64)
    a = (a @ a.T + 10 * torch.eye(10, dtype=torch.float64)).requires_grad_()
    ag = a.detach().to(dev).requires_grad_()
    def f(x):
        return torch.linalg.cholesky(x).sum() + torch.linalg.eigvalsh(x).pow(2).sum() + torch.linalg.inv(x).trace() \
            + torch.linalg.svdvals(x).sum()
    f(a).backward()
    f(ag).backward()
    close(ag.grad, a.grad, 1e-8)


@case("non-symmetric eigen decomposition")
def _():
    for dtype, tol in DTYPES:
        for shape in [(6, 6), (3, 15, 15), (120, 120)]:
            a = matrices(shape, dtype)
            w, v = torch.linalg.eig(a.to(dev))
            w_cpu = torch.linalg.eigvals(a)
            # the same eigenvalues (in any order)
            key = lambda x: x.real * 1000 + x.imag
            close(torch.sort(key(w.cpu()), dim=-1).values, torch.sort(key(w_cpu), dim=-1).values, tol * 100)
            ac = a.to(dev).to(v.dtype)
            close(ac @ v, v * w.unsqueeze(-2), tol * 100)
            close(v.abs().pow(2).sum(-2), torch.ones(w.shape, dtype=torch.float64), tol * 10)
    # complex conjugate pairs of a rotation
    r = torch.tensor([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    w = sorted(torch.linalg.eigvals(r.to(dev)).cpu().tolist(), key=lambda x: (round(x.real, 3), x.imag))
    assert all(abs(x - y) < 1e-5 for x, y in zip(w, [-2j, 2j, 3 + 0j])), f"eigenvalues {w}"


fails = [(n, m) for n, m in results if m is not None]
print(f"\n{len(results) - len(fails)} of {len(results)} passed")
for name, msg in fails:
    print(f"  {name}: {msg}")
sys.exit(0 if all(name in EXPECTED_FAILURES for name, _ in fails) else 1)
