"""torch.distributed with the NCCL backend (LibreCUDA's NCCL) on two GPUs, each result compared with the CPU, and
DistributedDataParallel training compared with training on one GPU. NCCL_TEST_WORLD=3 / 4 runs 3 / 4 ranks on the two
GPUs (LibreCUDA's NCCL only: NVIDIA's refuses several ranks on one GPU):

    L=build/lib
    CUDA_VISIBLE_DEVICES=0,1 \\
    LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1:$L/libcublasLt.so.13:$L/libcublas.so.13:$L/libcudnn.so.9:$L/libnccl.so.2 \\
        python3 tests/pytorch/nccl_ops.py [-v]

Without LD_PRELOAD the script runs on NVIDIA's CUDA and NCCL.
"""
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ranks (NCCL_TEST_WORLD, default 2), spread over the GPUs (several ranks on one GPU when there are more)
WORLD = int(os.environ.get("NCCL_TEST_WORLD", "2"))


def close(a, b, tol, what=""):
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    err = (a - b).abs().max().item() if a.numel() else 0
    scale = max(b.abs().max().item() if b.numel() else 1, 1)
    assert err <= tol * scale, f"{what}: max diff {err:.3g} (scale {scale:.3g})"


def inputs(shape, dtype, seed):
    """the input of each rank (the same on every rank, so each can compute the expected results)"""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(WORLD):
        if dtype.is_floating_point:
            out.append(torch.randn(shape, generator=g).to(dtype))
        else:
            out.append(torch.randint(-50, 50, shape, generator=g).to(dtype))
    return out


def collectives(rank, dev):
    for dtype, tol in [(torch.float32, 1e-6), (torch.float64, 1e-12), (torch.float16, 2e-3), (torch.bfloat16, 1e-2),
                       (torch.int32, 0), (torch.int64, 0), (torch.uint8, 0)]:
        for numel in [1, 1000, 5_000_001]:  # (the last spans several 16 MB chunks)
            if dtype == torch.uint8:
                xs = [x.abs() % 7 for x in inputs((numel,), torch.int32, numel)]
                xs = [x.to(dtype) for x in xs]
            else:
                xs = inputs((numel,), dtype, numel)
            mine = xs[rank].to(dev)
            ops = [(dist.ReduceOp.SUM, sum(x.double() for x in xs)), (dist.ReduceOp.MAX, torch.stack(xs).max(0)[0]),
                   (dist.ReduceOp.MIN, torch.stack(xs).min(0)[0])]
            # (products of small integers, within the type's range)
            if numel < 1000_000 and (dtype != torch.uint8 or 6 ** WORLD < 256):
                ops.append((dist.ReduceOp.PRODUCT, torch.stack([x.double() for x in xs]).prod(0)))
            if dtype.is_floating_point:
                ops.append((dist.ReduceOp.AVG, sum(x.double() for x in xs) / WORLD))
            for op, expected in ops:
                t = mine.clone()
                dist.all_reduce(t, op=op)
                close(t, expected.to(dtype), tol, f"all_reduce {op} {dtype} {numel}")
            t = mine.clone()
            dist.broadcast(t, src=1)
            close(t, xs[1], 0, f"broadcast {dtype}")
            t = mine.clone()
            dist.reduce(t, dst=0)
            if rank == 0:
                close(t, sum(x.double() for x in xs).to(dtype), tol, f"reduce {dtype}")
            out = torch.empty(WORLD * numel, dtype=dtype, device=dev)
            dist.all_gather_into_tensor(out, mine)
            close(out, torch.cat(xs), 0, f"all_gather {dtype}")
            if numel % WORLD == 0:
                out = torch.empty(numel // WORLD, dtype=dtype, device=dev)
                dist.reduce_scatter_tensor(out, mine)
                expected = sum(x.double() for x in xs).to(dtype).chunk(WORLD)[rank]
                close(out, expected, tol, f"reduce_scatter {dtype}")
                out = torch.empty_like(mine)
                dist.all_to_all_single(out, mine)
                expected = torch.cat([x.chunk(WORLD)[rank] for x in xs])
                close(out, expected, 0, f"all_to_all {dtype}")


def point_to_point(rank, dev):
    xs = inputs((3_000_001,), torch.float32, 7)
    t = xs[rank].to(dev)
    # around the ring: blocking, rank 0 sends first
    right, left = (rank + 1) % WORLD, (rank - 1) % WORLD
    r = torch.empty_like(t)
    if rank == 0:
        dist.send(t, right)
        dist.recv(r, left)
    else:
        dist.recv(r, left)
        dist.send(t, right)
    close(r, xs[left], 0, "send/recv")
    # both directions at once
    r = torch.empty_like(t)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, t, right), dist.P2POp(dist.irecv, r, left)])
    for q in reqs:
        q.wait()
    close(r, xs[left], 0, "batch_isend_irecv")
    # lists
    out = [torch.empty(5, device=dev) for _ in range(WORLD)]
    dist.all_gather(out, torch.full((5,), float(rank), device=dev))
    close(torch.stack(out), torch.arange(WORLD).float()[:, None].expand(WORLD, 5), 0, "all_gather list")
    objs = [None] * WORLD
    dist.all_gather_object(objs, {"rank": rank})
    assert objs == [{"rank": r} for r in range(WORLD)], objs


def groups(rank, dev):
    # a group of ranks 0 and 1, and one of only this rank (communicator splits)
    pair = dist.new_group([0, 1])
    if rank < 2:
        t = torch.full((10,), float(rank + 1), device=dev)
        dist.all_reduce(t, group=pair)
        close(t, torch.full((10,), 3.0), 0, "new_group all_reduce")
    alone = [dist.new_group([r]) for r in range(WORLD)][rank]
    t = torch.full((10,), float(rank + 1), device=dev)
    dist.all_reduce(t, group=alone)
    close(t, torch.full((10,), float(rank + 1)), 0, "single rank group")
    dist.barrier()


def librecuda_runtime():
    with open("/proc/self/maps") as f:
        return any("libcudart.so" in line and "/nvidia/" not in line for line in f)


def graphs(rank, dev):
    """collectives captured in a CUDA graph (torch.cuda.graphs), replayed with new inputs, eager ones in between"""
    if WORLD > torch.cuda.device_count() and librecuda_runtime():
        # (LibreCUDA's NCCL refuses: the runtime doesn't preempt a running kernel for another process's)
        return
    sizes = [1000, 3_000_001]  # one small exchange (below 64 KB), one of several chunks (12 MB)
    ins = {n: torch.zeros(n, device=dev) for n in sizes}
    reduced = {n: torch.zeros(n, device=dev) for n in sizes}
    cast = torch.zeros(2_000_000, device=dev)
    gathered = torch.zeros(WORLD * 1000, device=dev)
    scattered = torch.zeros(1000, device=dev)
    big_gathered = torch.zeros(WORLD * 1_000_000, device=dev)

    def body():
        for n in sizes:
            reduced[n].copy_(ins[n])
            dist.all_reduce(reduced[n])
        cast.copy_(ins[sizes[1]][:cast.numel()])
        dist.broadcast(cast, src=WORLD - 1)
        dist.all_gather_into_tensor(gathered, ins[sizes[0]])
        dist.all_gather_into_tensor(big_gathered, ins[sizes[1]][:1_000_000])
        dist.reduce_scatter_tensor(scattered, ins[sizes[1]][:WORLD * 1000])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()  # (warm-up outside the capture, as torch.cuda.graphs asks)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for replay in range(6):
        xs = {n: inputs((n,), torch.float32, 100 * replay + n) for n in sizes}
        for n in sizes:
            ins[n].copy_(xs[n][rank])
        g.replay()
        if replay % 2 == 1:
            # an eager collective between replays
            e = torch.full((5000,), float(rank + replay), device=dev)
            dist.all_reduce(e)
            close(e, torch.full((5000,), float(sum(range(WORLD)) + WORLD * replay)), 0, f"eager all_reduce {replay}")
        for n in sizes:
            close(reduced[n], sum(x.double() for x in xs[n]).float(), 1e-5, f"graph all_reduce {n} replay {replay}")
        big = xs[sizes[1]]
        close(cast, big[WORLD - 1][:cast.numel()], 0, f"graph broadcast replay {replay}")
        close(gathered, torch.cat(xs[sizes[0]]), 0, f"graph all_gather replay {replay}")
        close(big_gathered, torch.cat([x[:1_000_000] for x in big]), 0, f"graph all_gather 4 MB replay {replay}")
        close(scattered, sum(x[:WORLD * 1000].double() for x in big).float().chunk(WORLD)[rank], 1e-5,
              f"graph reduce_scatter replay {replay}")


def model():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                               torch.nn.Flatten(), torch.nn.Linear(8 * 8 * 8, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 10))


def data(steps):
    g = torch.Generator().manual_seed(2)
    # (24 images: the same number per rank for 2, 3, 4 ranks)
    return [(torch.randn(24, 3, 8, 8, generator=g), torch.randint(0, 10, (24,), generator=g)) for _ in range(steps)]


def ddp(rank, dev):
    steps = 5
    m = torch.nn.parallel.DistributedDataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model()).to(dev),
                                                  device_ids=[dev.index], bucket_cap_mb=0.01)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    for x, y in data(steps):
        x, y = x.chunk(WORLD)[rank].to(dev), y.chunk(WORLD)[rank].to(dev)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt.step()
    # the same training on one GPU with the whole batches
    ref = model().to(dev)
    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9)
    for x, y in data(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(ref(x.to(dev)), y.to(dev)).backward()
        opt.step()
    for (name, a), b in zip(m.module.state_dict().items(), ref.state_dict().values()):
        # (more ranks sum the gradients in other orders)
        close(a, b, 1e-4 * max(1, WORLD - 1), f"DDP {name}")


CASES = [("all-reduce, broadcast, reduce, all-gather, reduce-scatter, all-to-all", collectives),
         ("send / receive, batched point-to-point, object gather", point_to_point),
         ("process groups (communicator splits)", groups),
         ("DistributedDataParallel with SyncBatchNorm vs one GPU", ddp),
         ("collectives in CUDA graphs, replayed", graphs)]
# NCCL_TEST_CASES=graphs,ddp: only the cases whose function names are listed
if os.environ.get("NCCL_TEST_CASES"):
    CASES = [c for c in CASES if c[1].__name__ in os.environ["NCCL_TEST_CASES"].split(",")]


def worker(rank, init, queue):
    dev = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_method=init, rank=rank, world_size=WORLD, device_id=dev)
    for name, fn in CASES:
        t0 = time.time()
        try:
            fn(rank, dev)
            torch.cuda.synchronize()
            queue.put((rank, name, None, time.time() - t0))
        except Exception as e:
            queue.put((rank, name, str(e).splitlines()[0][:300], time.time() - t0))
            if "-v" in sys.argv:
                traceback.print_exc()
        dist.barrier()
    dist.destroy_process_group()


def single_process():
    """torch.cuda.nccl: one process, both GPUs (ncclCommInitAll and grouped collectives)"""
    import torch.cuda.nccl as nccl
    xs = inputs((1000,), torch.float32, 3)[:2]
    ts = [x.cuda(d) for d, x in enumerate(xs)]
    nccl.all_reduce(ts)
    for t in ts:
        close(t, xs[0] + xs[1], 1e-6, "torch.cuda.nccl.all_reduce")
    ts = [x.cuda(d) for d, x in enumerate(xs)]
    nccl.broadcast(ts)  # (from the first tensor)
    for t in ts:
        close(t, xs[0], 0, "torch.cuda.nccl.broadcast")


def main():
    os.environ.setdefault("TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false")
    if torch.cuda.device_count() < 2:
        print("needs 2 GPUs")
        sys.exit(1)
    failures = 0
    t0 = time.time()
    try:
        single_process()
        print(f"ok    torch.cuda.nccl in one process ({time.time() - t0:.2f}s)", flush=True)
    except Exception as e:
        failures += 1
        print(f"FAIL  torch.cuda.nccl in one process: {str(e).splitlines()[0][:300]}", flush=True)
        if "-v" in sys.argv:
            traceback.print_exc()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = "file://" + os.path.join(d, "init")
        procs = [ctx.Process(target=worker, args=(r, init, queue)) for r in range(WORLD)]
        for p in procs:
            p.start()
        for _ in range(len(CASES)):
            try:
                reports = [queue.get(timeout=float(os.environ.get("NCCL_TEST_TIMEOUT", "300"))) for _ in range(WORLD)]
            except Exception:
                print("FAIL  timed out (a rank hangs)", flush=True)
                for p in procs:
                    p.kill()
                sys.exit(1)
            name, errors = reports[0][1], [f"rank {r}: {e}" for r, _, e, _ in reports if e]
            elapsed = max(t for *_, t in reports)
            if errors:
                failures += 1
                print(f"FAIL  {name}: {'; '.join(errors)}", flush=True)
            else:
                print(f"ok    {name} ({elapsed:.2f}s)", flush=True)
        for p in procs:
            p.join()
            failures += p.exitcode != 0
    print(f"{len(CASES) + 1 - failures} of {len(CASES) + 1} passed" if failures == 0 else f"{failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
