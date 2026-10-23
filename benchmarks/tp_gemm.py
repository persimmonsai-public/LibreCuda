"""Tensor-parallel GEMM on two GPUs: a row-parallel layer Y = X @ W with K split between the ranks; each rank
multiplies its half (bf16) and the partial results are all-reduced. Sequential (GEMM, then all-reduce) and chunked
overlap (the all-reduce of a row block runs while the next one is multiplied), checked against a float64 reference.
Compares NCCL implementations: run it with NVIDIA's NCCL, or with LibreCUDA's (on NVIDIA's runtime by preloading
only build/lib/libnccl.so.2 -- copied next to NVIDIA's libcudart.so.13 -- or on LibreCUDA's whole stack).

    CUDA_VISIBLE_DEVICES=0,1 python3 benchmarks/tp_gemm.py "NVIDIA NCCL"
    CUDA_VISIBLE_DEVICES=0,1 LD_PRELOAD=build/lib/libcudart.so.13:build/lib/libcuda.so.1:build/lib/libcublasLt.so.13:\
        build/lib/libcublas.so.13:build/lib/libnccl.so.2 python3 benchmarks/tp_gemm.py LibreCUDA
"""
import os, sys, time, tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp

SHAPES = [(512, 8192, 4096), (2048, 8192, 4096), (8192, 8192, 4096)]  # M, K, N

def layer_seq(x, w):
    y = x @ w
    dist.all_reduce(y)
    return y

def layer_overlap(x, w, chunks):
    rows = x.shape[0] // chunks
    y = torch.empty(x.shape[0], w.shape[1], dtype=x.dtype, device=x.device)
    works = []
    for i in range(chunks):
        torch.matmul(x[i * rows:(i + 1) * rows], w, out=y[i * rows:(i + 1) * rows])
        works.append(dist.all_reduce(y[i * rows:(i + 1) * rows], async_op=True))
    for wk in works:
        wk.wait()
    return y

def timeit(fn, reps):
    for _ in range(2): fn()
    torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

def worker(rank, init, results):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    out = []
    for M, K, N in SHAPES:
        g = torch.Generator().manual_seed(0)
        X = torch.randn(M, K, generator=g) / K ** 0.25
        W = torch.randn(K, N, generator=g) / K ** 0.25
        ref = (X.double() @ W.double()).float()
        h = K // 2
        x = X[:, rank * h:(rank + 1) * h].to("cuda", torch.bfloat16).contiguous()
        w = W[rank * h:(rank + 1) * h].to("cuda", torch.bfloat16).contiguous()
        reps = 20 if M <= 2048 else 8
        row = {"shape": (M, K, N), "gemm": timeit(lambda: x @ w, reps)}
        y = layer_seq(x, w)
        row["err_seq"] = (y.float().cpu() - ref).abs().max().item()
        row["seq"] = timeit(lambda: layer_seq(x, w), reps)
        for chunks in (4, 8):
            y = layer_overlap(x, w, chunks)
            row[f"err_ov{chunks}"] = (y.float().cpu() - ref).abs().max().item()
            row[f"ov{chunks}"] = timeit(lambda: layer_overlap(x, w, chunks), reps)
        out.append(row)
    if rank == 0:
        results.extend(out)
    dist.destroy_process_group()

if __name__ == "__main__":
    d = tempfile.mkdtemp()
    with mp.Manager() as m:
        results = m.list()
        mp.spawn(worker, args=("file://" + d + "/i", results), nprocs=2)
        label = sys.argv[1] if len(sys.argv) > 1 else ""
        for r in results:
            M, K, N = r["shape"]
            mb = M * N * 2 / 2 ** 20
            print(f"{label:24s} M={M:5d} K={K} N={N} ({mb:4.0f} MB out): GEMM {r['gemm']:6.2f} ms | "
                  f"seq {r['seq']:6.2f} | overlap4 {r['ov4']:6.2f} | overlap8 {r['ov8']:6.2f} ms | "
                  f"max err {max(r['err_seq'], r['err_ov4'], r['err_ov8']):.3f}", flush=True)
