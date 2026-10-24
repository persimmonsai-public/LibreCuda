"""Decode-sized row-parallel layers on two GPUs: Y = X @ W in bf16, X M x 8192, W 8192 x 4096 split along K (each
rank multiplies its half, then the M x 4096 partial results are all-reduced), for M = 1 to 128. Per M: the GEMM alone,
the all-reduce alone, GEMM then all-reduce, and the same captured in a CUDA graph (20 layers per graph; NO_GRAPH=1
skips it). Run like benchmarks/tp_gemm.py:

    CUDA_VISIBLE_DEVICES=0,1 python3 benchmarks/tp_small.py "NVIDIA NCCL"
    L=build/lib CUDA_VISIBLE_DEVICES=0,1 \\
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libnccl.so.2 \\
        python3 benchmarks/tp_small.py LibreCUDA
"""
import os, sys, time, tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp

MS = [1, 8, 32, 128]
K, N = 8192, 4096

def timeit(fn, reps):
    for _ in range(20): fn()
    torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e6

def worker(rank, init, results):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=2, device_id=torch.device("cuda", rank))
    out = []
    w = (torch.randn(K // 2, N, device="cuda") / 50).to(torch.bfloat16)
    for M in MS:
        x = (torch.randn(M, K // 2, device="cuda") / 50).to(torch.bfloat16)
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        reps = 2000
        row = {"M": M}
        row["gemm"] = timeit(lambda: torch.matmul(x, w, out=y), reps)
        row["ar"] = timeit(lambda: dist.all_reduce(y), reps)
        def layer():
            torch.matmul(x, w, out=y)
            dist.all_reduce(y)
        row["layer"] = timeit(layer, reps)
        try:
            if os.environ.get("NO_GRAPH"): raise RuntimeError("skipped")
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): layer()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize(); dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(20): layer()
            row["graph"] = timeit(g.replay, reps // 20) / 20
        except Exception as e:
            row["graph"] = float("nan")
            print(f"rank {rank} graph failed: {e}", file=sys.stderr)
        out.append(row)
    if rank == 0: results.extend(out)
    dist.destroy_process_group()

if __name__ == "__main__":
    d = tempfile.mkdtemp()
    with mp.Manager() as m:
        results = m.list()
        mp.spawn(worker, args=("file://" + d + "/i", results), nprocs=2)
        label = sys.argv[1] if len(sys.argv) > 1 else ""
        for r in results:
            print(f"{label:28s} M={r['M']:4d} ({r['M'] * N * 2 >> 10:4d} KB): GEMM {r['gemm']:7.1f} us | all-reduce "
                  f"{r['ar']:6.1f} us | layer {r['layer']:7.1f} us | graph {r['graph']:7.1f} us", flush=True)
