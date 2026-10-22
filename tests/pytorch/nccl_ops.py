"""torch.distributed with the NCCL backend (LibreCUDA's NCCL) on two GPUs, each result compared with the CPU, and
DistributedDataParallel training compared with training on one GPU:

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

WORLD = 2


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
            if numel < 1000_000:
                ops.append((dist.ReduceOp.PRODUCT, xs[0].double() * xs[1].double()))
            if dtype.is_floating_point:
                ops.append((dist.ReduceOp.AVG, (xs[0].double() + xs[1].double()) / 2))
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
                close(t, (xs[0].double() + xs[1].double()).to(dtype), tol, f"reduce {dtype}")
            out = torch.empty(WORLD * numel, dtype=dtype, device=dev)
            dist.all_gather_into_tensor(out, mine)
            close(out, torch.cat(xs), 0, f"all_gather {dtype}")
            if numel % WORLD == 0:
                out = torch.empty(numel // WORLD, dtype=dtype, device=dev)
                dist.reduce_scatter_tensor(out, mine)
                expected = (xs[0].double() + xs[1].double()).to(dtype).chunk(WORLD)[rank]
                close(out, expected, tol, f"reduce_scatter {dtype}")
                out = torch.empty_like(mine)
                dist.all_to_all_single(out, mine)
                expected = torch.cat([x.chunk(WORLD)[rank] for x in xs])
                close(out, expected, 0, f"all_to_all {dtype}")


def point_to_point(rank, dev):
    xs = inputs((3_000_001,), torch.float32, 7)
    t = xs[rank].to(dev)
    peer = 1 - rank
    # blocking: rank 0 sends first
    if rank == 0:
        dist.send(t, 1)
        r = torch.empty_like(t)
        dist.recv(r, 1)
    else:
        r = torch.empty_like(t)
        dist.recv(r, 0)
        dist.send(t, 0)
    close(r, xs[peer], 0, "send/recv")
    # both directions at once
    r = torch.empty_like(t)
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, t, peer), dist.P2POp(dist.irecv, r, peer)])
    for q in reqs:
        q.wait()
    close(r, xs[peer], 0, "batch_isend_irecv")
    # lists
    out = [torch.empty(5, device=dev) for _ in range(WORLD)]
    dist.all_gather(out, torch.full((5,), float(rank), device=dev))
    close(torch.stack(out), torch.arange(WORLD).float()[:, None].expand(WORLD, 5), 0, "all_gather list")
    objs = [None] * WORLD
    dist.all_gather_object(objs, {"rank": rank})
    assert objs == [{"rank": r} for r in range(WORLD)], objs


def groups(rank, dev):
    # a group with both ranks, and one of only this rank (communicator splits)
    both = dist.new_group([0, 1])
    t = torch.full((10,), float(rank + 1), device=dev)
    dist.all_reduce(t, group=both)
    close(t, torch.full((10,), 3.0), 0, "new_group all_reduce")
    alone = [dist.new_group([r]) for r in range(WORLD)][rank]
    t = torch.full((10,), float(rank + 1), device=dev)
    dist.all_reduce(t, group=alone)
    close(t, torch.full((10,), float(rank + 1)), 0, "single rank group")
    dist.barrier()


def model():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                               torch.nn.Flatten(), torch.nn.Linear(8 * 8 * 8, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 10))


def data(steps):
    g = torch.Generator().manual_seed(2)
    return [(torch.randn(16, 3, 8, 8, generator=g), torch.randint(0, 10, (16,), generator=g)) for _ in range(steps)]


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
        close(a, b, 1e-4, f"DDP {name}")


CASES = [("all-reduce, broadcast, reduce, all-gather, reduce-scatter, all-to-all", collectives),
         ("send / receive, batched point-to-point, object gather", point_to_point),
         ("process groups (communicator splits)", groups),
         ("DistributedDataParallel with SyncBatchNorm vs one GPU", ddp)]


def worker(rank, init, queue):
    dev = torch.device("cuda", rank)
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
    xs = inputs((1000,), torch.float32, 3)
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
    if torch.cuda.device_count() < WORLD:
        print(f"needs {WORLD} GPUs")
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
    queue = ctx.SimpleQueue()
    with tempfile.TemporaryDirectory() as d:
        init = "file://" + os.path.join(d, "init")
        procs = [ctx.Process(target=worker, args=(r, init, queue)) for r in range(WORLD)]
        for p in procs:
            p.start()
        for _ in range(len(CASES)):
            reports = [queue.get() for _ in range(WORLD)]
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
