"""Sparse tensors on the GPU (cuSPARSE), each compared with the dense result on the CPU:

    L=build/lib
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcusolver.so.12:\\
$L/libcusparse.so.12 python3 tests/pytorch/sparse_ops.py [-v]

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


def close(a, b, tol):
    a = a.to_dense() if a.layout != torch.strided else a
    b = b.to_dense() if b.layout != torch.strided else b
    wide = lambda x: x.cpu().to(torch.complex128 if x.is_complex() else torch.float64)
    a, b = wide(a), wide(b)
    err = (a - b).abs().max().item() if a.numel() else 0
    scale = max(b.abs().max().item() if b.numel() else 1, 1)
    assert err <= tol * scale, f"max diff {err:.3g} (scale {scale:.3g})"


DTYPES = [(torch.float32, 1e-4), (torch.float64, 1e-10), (torch.complex64, 1e-4), (torch.complex128, 1e-10)]


def sparse_dense(shape, dtype, density=0.2):
    """a dense tensor with roughly `density` of its entries nonzero"""
    a = torch.randn(*shape, dtype=dtype)
    return a * (torch.rand(*shape) < density)


@case("COO @ dense (torch.sparse.mm), COO @ vector")
def _():
    for dtype, tol in DTYPES:
        a = sparse_dense((70, 50), dtype)
        b = torch.randn(50, 33, dtype=dtype)
        s = a.to_sparse().to(dev)
        close(torch.sparse.mm(s, b.to(dev)), a @ b, tol)
        close(s @ b[:, 0].to(dev), a @ b[:, 0], tol)
        # addmm with a sparse matrix
        c = torch.randn(70, 33, dtype=dtype)
        close(torch.sparse.addmm(c.to(dev), s, b.to(dev), beta=0.5, alpha=2.0), 0.5 * c + 2.0 * (a @ b), tol)


@case("CSR @ dense, CSR @ vector, addmm / addmv, transposed operands")
def _():
    for dtype, tol in DTYPES:
        for idx in [torch.int32, torch.int64]:
            a = sparse_dense((64, 48), dtype)
            b = torch.randn(48, 20, dtype=dtype)
            s = a.to_sparse_csr().to(dev)
            if idx == torch.int32:
                s = torch.sparse_csr_tensor(s.crow_indices().int(), s.col_indices().int(), s.values(), s.shape)
            close(s @ b.to(dev), a @ b, tol)
            close(s @ b[:, 3].to(dev), a @ b[:, 3], tol)
            c = torch.randn(64, 20, dtype=dtype)
            close(torch.addmm(c.to(dev), s, b.to(dev), beta=-1.0, alpha=0.5), -c + 0.5 * (a @ b), tol)
            v = torch.randn(64, dtype=dtype)
            close(torch.addmv(v.to(dev), s, b[:, 1].to(dev), beta=2.0), 2 * v + a @ b[:, 1], tol)
            # dense @ CSR (the sparse operand transposed), row-major output
            d = torch.randn(10, 64, dtype=dtype)
            close(d.to(dev) @ s, d @ a, tol)


@case("half and bfloat16 CSR @ dense")
def _():
    for dtype in [torch.float16, torch.bfloat16]:
        a = sparse_dense((96, 64), torch.float32).to(dtype)
        b = torch.randn(64, 40).to(dtype)
        got = (a.to_sparse_csr().to(dev) @ b.to(dev)).float()
        close(got, a.float() @ b.float(), 2e-2)


@case("CSR @ CSR, COO @ COO (SpGEMM)")
def _():
    for dtype, tol in DTYPES:
        a = sparse_dense((60, 45), dtype, 0.1)
        b = sparse_dense((45, 50), dtype, 0.1)
        got = a.to_sparse_csr().to(dev) @ b.to_sparse_csr().to(dev)
        assert got.layout == torch.sparse_csr, got.layout
        close(got, a @ b, tol)
        got = torch.sparse.mm(a.to_sparse().to(dev), b.to_sparse().to(dev))
        close(got, a @ b, tol)


@case("CSR + CSR (csrgeam2)")
def _():
    for dtype, tol in DTYPES:
        a = sparse_dense((40, 35), dtype)
        b = sparse_dense((40, 35), dtype)
        got = torch.add(a.to_sparse_csr().to(dev), b.to_sparse_csr().to(dev), alpha=-2.0)
        close(got, a - 2.0 * b, tol)


@case("sampled_addmm (SDDMM)")
def _():
    for dtype, tol in DTYPES:
        mask = sparse_dense((31, 40), dtype)
        m1 = torch.randn(31, 17, dtype=dtype)
        m2 = torch.randn(17, 40, dtype=dtype)
        got = torch.sparse.sampled_addmm(mask.to_sparse_csr().to(dev), m1.to(dev), m2.to(dev), beta=0.5, alpha=2.0)
        want = (0.5 * mask + 2.0 * (m1 @ m2)) * (mask != 0)
        close(got, want, tol * 10)


@case("triangular solves with sparse CSR (SpSV, SpSM)")
def _():
    for dtype, tol in DTYPES:
        n = 80
        a = sparse_dense((n, n), dtype, 0.1) + 4 * torch.eye(n, dtype=dtype)
        for upper in [False, True]:
            t = a.triu() if upper else a.tril()
            s = t.to_sparse_csr().to(dev)
            for unit in [False, True]:
                b = torch.randn(n, 7, dtype=dtype)
                got = torch.triangular_solve(b.to(dev), s, upper=upper, unitriangular=unit).solution
                want = torch.linalg.solve_triangular(t, b, upper=upper, unitriangular=unit)
                close(got, want, tol * 100)
                # one right-hand side
                got = torch.triangular_solve(b[:, :1].to(dev), s, upper=upper).solution
                close(got, torch.linalg.solve_triangular(t, b[:, :1], upper=upper), tol * 100)
            # transposed
            b = torch.randn(n, 4, dtype=dtype)
            got = torch.triangular_solve(b.to(dev), s, upper=upper, transpose=True).solution
            close(got, torch.linalg.solve_triangular(t.mT, b, upper=not upper), tol * 100)


@case("BSR @ dense, BSR @ vector, triangular solves with BSR")
def _():
    for dtype, tol in DTYPES:
        blocks = (3, 3)
        a = torch.zeros(27, 27, dtype=dtype)
        for i in range(9):
            for j in range(9):
                if i == j or (i * 7 + j * 3) % 5 == 0:
                    a[3 * i:3 * i + 3, 3 * j:3 * j + 3] = torch.randn(3, 3, dtype=dtype)
        a += 6 * torch.eye(27, dtype=dtype)
        s = a.to_sparse_bsr(blocks).to(dev)
        b = torch.randn(27, 8, dtype=dtype)
        close(s @ b.to(dev), a @ b, tol)
        close(s @ b[:, 0].to(dev), a @ b[:, 0], tol)
        for upper in [False, True]:
            t = a.triu() if upper else a.tril()
            st = t.to_sparse_bsr(blocks).to(dev)
            got = torch.triangular_solve(b.to(dev), st, upper=upper).solution
            close(got, torch.linalg.solve_triangular(t, b, upper=upper), tol * 100)


@case("conversions between dense, COO, CSR, CSC")
def _():
    for dtype, _tol in DTYPES:
        a = sparse_dense((25, 33), dtype)
        g = a.to(dev)
        close(g.to_sparse(), a, 0)
        close(g.to_sparse_csr(), a, 0)
        close(g.to_sparse_csc(), a, 0)
        close(g.to_sparse_csr().to_sparse_csc(), a, 0)
        close(g.to_sparse().to_sparse_csr(), a, 0)
        close(g.to_sparse_csr().to_sparse_coo().coalesce(), a, 0)


@case("autograd through sparse.mm")
def _():
    a = sparse_dense((40, 30), torch.float64)
    s = a.to_sparse().to(dev).requires_grad_(True)
    w = torch.randn(30, 6, dtype=torch.float64, device=dev, requires_grad=True)
    (torch.sparse.mm(s, w) ** 2).sum().backward()
    out = a @ w.detach().cpu()
    close(w.grad, a.t() @ (2 * out), 1e-10)
    # the gradient of the sparse operand at its pattern
    close(s.grad.to_dense() * (a != 0).to(dev), ((2 * out) @ w.detach().cpu().t()) * (a != 0), 1e-10)


fails = [(n, m) for n, m in results if m is not None]
print(f"\n{len(results) - len(fails)} of {len(results)} passed")
for name, msg in fails:
    print(f"  {name}: {msg}")
sys.exit(0 if not fails else 1)
