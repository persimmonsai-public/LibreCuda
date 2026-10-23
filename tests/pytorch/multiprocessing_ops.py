"""CUDA tensors shared between processes with torch.multiprocessing (CUDA IPC: memory and event handles):

    L=build/lib
    CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=$L/libcudart.so.13:$L/libcuda.so.1 python3 tests/pytorch/multiprocessing_ops.py

Without LD_PRELOAD the script runs on NVIDIA's CUDA.
"""
import sys
import time

import torch
import torch.multiprocessing as mp


def consumer(inbox, outbox):
    torch.cuda.init()
    for _ in range(3):
        name, tensors = inbox.get()
        # in place: the producer sees the changes
        for t in tensors:
            t.mul_(-2)
        torch.cuda.synchronize()
        outbox.put((name, [float(t.sum()) for t in tensors]))
        del tensors
    outbox.put("done")


def main():
    ctx = mp.get_context("spawn")
    inbox, outbox = ctx.Queue(), ctx.Queue()
    process = ctx.Process(target=consumer, args=(inbox, outbox))
    process.start()
    failures = 0
    cases = []
    # a tensor written by a kernel just before it is sent (the consumer waits for the producer's event)
    a = torch.arange(1000, device="cuda", dtype=torch.float32)
    cases.append(("tensor written just before sending", [a], [-2 * a.sum().item()]))
    # two views of one storage (the same memory opened once per process), with offsets
    base = torch.ones(4096, device="cuda")
    views = [base[100:200], base[3000:3500]]
    cases.append(("views of one storage", views, [-200.0, -1000.0]))
    big = torch.full((64, 1024, 1024), 0.5, device="cuda")
    cases.append(("256 MB tensor", [big], [-2 * 0.5 * big.numel()]))
    expected_after = {
        "tensor written just before sending": lambda: torch.equal(a, -2 * torch.arange(1000, device="cuda",
                                                                                          dtype=torch.float32)),
        "views of one storage": lambda: base[100:200].eq(-2).all().item() and base[3000:3500].eq(-2).all().item()
                                        and base[:100].eq(1).all().item(),
        "256 MB tensor": lambda: big.eq(-1).all().item(),
    }
    for name, tensors, sums in cases:
        t0 = time.time()
        inbox.put((name, tensors))
        got_name, got = outbox.get(timeout=120)
        torch.cuda.synchronize()
        ok = got_name == name and all(abs(g - s) <= 1e-3 * max(1, abs(s)) for g, s in zip(got, sums))
        ok = ok and expected_after[name]()
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {name} ({time.time() - t0:.2f}s){'' if ok else f': {got}'}", flush=True)
    assert outbox.get(timeout=60) == "done"
    process.join(60)
    failures += process.exitcode != 0
    print(f"{3 - failures} of 3 passed" if failures == 0 else f"{failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
