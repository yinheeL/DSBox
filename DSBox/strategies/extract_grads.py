# file: selectiontools/strategies/extract_grads.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _flat(gs: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.detach().reshape(-1) for g in gs]).to(torch.float32)


def _param_dim(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def extract_gradients(
    model: torch.nn.Module,
    dataset,
    loss_fn: Callable,
    grad_dir: Union[str, Path],
    batch_size: int = 8,
    device: Union[str, torch.device, None] = None,
    proj_dim: int | None = None,
    proj_block_size: int = 1000,
):
    """
    Extract the gradient of each sample on the dataset, and optionally project it to lower dimensions in real time:
    - When proj_dim=None, write grads.npy & labels.npy
    - When proj_dim=int, write grads_proj.npy & labels.npy
    Added batch-level progress printing.
    """
    grad_dir = Path(grad_dir)
    grad_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()

    N = len(dataset)
    D = _param_dim(model)
    print(f"[extract_grads] START N={N}, D={D}, proj_dim={proj_dim}, batch_size={batch_size}")

    # labels memmap
    label_path = grad_dir / "labels.npy"
    y_mm = np.memmap(label_path, dtype="int64", mode="w+", shape=(N,))

    # grads memmap
    if proj_dim is None:
        grad_path = grad_dir / "grads.npy"
        g_mm = np.memmap(grad_path, dtype="float32", mode="w+", shape=(N, D))
    else:
        grad_path = grad_dir / "grads_proj.npy"
        g_mm = np.memmap(grad_path, dtype="float32", mode="w+", shape=(N, proj_dim))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    params = [p for p in model.parameters() if p.requires_grad]

    idx = 0
    log_interval = max(1, len(loader) // 10)
    for batch_idx, batch in enumerate(loader):

        if isinstance(batch, dict):
            ys = batch.get("labels", batch.get("label")).to(device)
            xs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "label")}
        elif isinstance(batch, (list, tuple)):
            *xs_parts, ys = batch
            ys = ys.to(device)
            if len(xs_parts) == 1:
                xs = xs_parts[0].to(device)
            else:
                xs = [t.to(device) for t in xs_parts]
        else:
            raise ValueError(f"Unrecognized batch type {type(batch)}")

        bs = ys.shape[0]
        for j in range(bs):

            if isinstance(xs, dict):
                inp_j = {k: v[j : j + 1] for k, v in xs.items()}
                out = model(**inp_j)
            elif isinstance(xs, (list, tuple)):
                inp_j = [t[j : j + 1] for t in xs]
                out = model(*inp_j)
            else:
                inp_j = xs[j : j + 1]
                out = model(inp_j)

            logits = out.logits if hasattr(out, "logits") else out[0]
            loss = loss_fn(logits, ys[j : j + 1])
            grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
            grads = [
                g if g is not None else torch.zeros_like(p, dtype=loss.dtype)
                for g, p in zip(grads, params)
            ]
            g_flat = _flat(grads).cpu().numpy()


            y_mm[idx] = ys[j].cpu().item()

            if proj_dim is None:
                g_mm[idx, :] = g_flat
            else:
                out_p = np.zeros((proj_dim,), dtype="float32")
                for k in range(0, D, proj_block_size):
                    end = min(D, k + proj_block_size)
                    blk = g_flat[k:end]
                    R_blk = np.random.randn(end - k, proj_dim).astype("float32")
                    out_p += blk.dot(R_blk)
                g_mm[idx, :] = out_p

            idx += 1


        if batch_idx % log_interval == 0:
            print(f"[extract_grads] batch {batch_idx+1}/{len(loader)} → processed {idx}/{N} samples")

    # flush
    g_mm.flush()
    y_mm.flush()
    print(f"[extract_grads] DONE processed {idx}/{N}")
