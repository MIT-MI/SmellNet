# evaluate.py
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, Optional, Sequence, Union, Tuple  # already present
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------ utils ------------------------
def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _maybe_to_device(x, device, dtype=None):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return type(x)(_maybe_to_device(t, device, dtype) for t in x)
    if torch.is_tensor(x):
        x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)  # <-- cast regardless of original dtype
        return x
    return x

def _unpack_batch(batch):
    # Supports (x, y) or (x, y, lengths)
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    x, y = batch
    return x, y, None

def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.shape[1])
    topk = torch.topk(logits, k=k, dim=1).indices  # (B, k)
    return (topk == targets.unsqueeze(1)).any(dim=1).sum().item()

def _build_class_to_category(
    class_names: Sequence[str],
    ingredient_to_category: Dict[str, str],
) -> Tuple[Dict[int, str], set]:
    """
    Returns:
      class_to_cat: {class_id -> category_name}
      missing: set of ingredient names not found in mapping
    """
    missing = set()
    class_to_cat: Dict[int, str] = {}
    for i, name in enumerate(class_names):
        cat = ingredient_to_category.get(name)
        if cat is None:
            missing.add(name)
            cat = "UNKNOWN"
        class_to_cat[i] = cat
    return class_to_cat, missing

# -------------------- classification -------------------
def evaluate(
    model: torch.nn.Module,
    data_loader,
    *,
    logger=None,
    topk: Sequence[int] = (1, 5),
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    logits_from_output: Optional[Callable[[object], torch.Tensor]] = None,
    ingredient_to_category: Optional[Dict[str, str]] = None,
    class_names: Optional[Sequence[str]] = None,   # usually le.classes_
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Generic classifier evaluation.
    Batches may be (x, y) or (x, y, lengths). Returns acc@k plus macro metrics.
    If your model returns a tuple (e.g., (logits, embedding)), pass a picker via logits_from_output.
    """
    dev = device or _device()
    model.to(dev).eval()

    # default: accept logits or (logits, ...)
    if logits_from_output is None:
        def logits_from_output(out):
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    total = 0
    correct_at = {k: 0 for k in topk}
    ys, preds = [], []
    # --- capture per-sample top-k hits if we need per-category acc@5 ---
    need_topk = any(k > 1 for k in topk)
    topk_hits_batches: Dict[int, list] = {k: [] for k in topk if k > 1}

    with torch.no_grad():
        for batch in data_loader:
            x, y, lengths = _unpack_batch(batch)
            x = _maybe_to_device(x, dev, dtype)
            y = _maybe_to_device(y, dev)
            lengths = _maybe_to_device(lengths, dev)

            out = model(x, lengths=lengths)
            logits = logits_from_output(out)

            bs = y.size(0)
            total += bs

            pred = logits.argmax(dim=1)
            ys.append(y.cpu())
            preds.append(pred.cpu())

            for k in topk:
                correct_at[k] += _topk_correct(logits, y, k)

            if need_topk:
                for k in topk:
                    if k > 1:
                        tk = torch.topk(logits, k=min(k, logits.shape[1]), dim=1).indices
                        hits = (tk == y.unsqueeze(1)).any(dim=1).cpu().numpy()  # (B,)
                        topk_hits_batches[k].append(hits)

    y_np = torch.cat(ys).numpy() if ys else np.array([])
    p_np = torch.cat(preds).numpy() if preds else np.array([])

    results: Dict[str, Union[float, np.ndarray, Dict]] = {}
    for k in topk:
        results[f"acc@{k}"] = 100.0 * (correct_at[k] / max(total, 1))

    # macro metrics
    if total > 0:
        results["precision_macro"] = precision_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["recall_macro"]    = recall_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["f1_macro"]        = f1_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["confusion_matrix"] = confusion_matrix(y_np, p_np)

    # --- per-category accuracy (optional) ---
    if (ingredient_to_category is not None) and (class_names is not None) and (len(y_np) > 0):
        class_to_cat, missing = _build_class_to_category(class_names, ingredient_to_category)
        if logger and missing:
            logger.warning(f"[evaluate] {len(missing)} classes missing in ingredient_to_category; counted as 'UNKNOWN'.")

        # true category per sample
        true_cat = np.array([class_to_cat[int(c)] for c in y_np], dtype=object)
        correct1 = (p_np == y_np)

        per_cat: Dict[str, Dict[str, float]] = {}
        for cat in np.unique(true_cat):
            mask = (true_cat == cat)
            n = int(mask.sum())
            acc1 = float(correct1[mask].mean()*100.0) if n > 0 else 0.0
            row = {"n": n, "acc@1": acc1}

            # acc@5 if requested
            if need_topk and 5 in topk:
                hits5 = np.concatenate(topk_hits_batches[5], axis=0) if topk_hits_batches[5] else np.array([])
                acc5 = float(hits5[mask].mean()*100.0) if hits5.size else 0.0
                row["acc@5"] = acc5

            per_cat[str(cat)] = row

        results["per_category"] = per_cat  # {category: {"n": int, "acc@1": %, "acc@5": %?}}

    if logger:
        msg = " | ".join([f"acc@{k}: {results[f'acc@{k}']:.2f}%" for k in topk])
        logger.info(f"✅ Evaluation — {msg}")
        if "per_category" in results:
            logger.info("Per-category acc:")
            for cat, row in results["per_category"].items():
                extras = f", acc@5={row['acc@5']:.2f}%" if ("acc@5" in row) else ""
                logger.info(f"  - {cat}: n={row['n']}, acc@1={row['acc@1']:.2f}%{extras}")
    else:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in results.items() if "matrix" not in k})

    return results

# --------------------- contrastive ---------------------
def evaluate_contrastive(
    gcms_encoder: torch.nn.Module,
    sensor_encoder: torch.nn.Module,
    *,
    gcms_data: Union[torch.Tensor, np.ndarray],      # (N_g, Dg) or tensor acceptable by encoder
    sensor_data: Union[torch.Tensor, np.ndarray],    # (N_s, T, C) or (N_s, Ds)
    sensor_labels: Union[torch.Tensor, np.ndarray],  # (N_s,) integer indices into gcms rows
    lengths: Optional[torch.Tensor] = None,          # optional for padded sensor sequences
    logger=None,
    l2_normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,                # set to chunk-encode large sensor_data
    ingredient_to_category: Optional[Dict[str, str]] = None,
    class_names: Optional[Sequence[str]] = None,     # le.classes_
    topk: Sequence[int] = (1, 5),                    # <<< align with `evaluate`
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Contrastive evaluation mirroring `evaluate`'s output format.
    Computes cosine similarity between sensor and GCMS embeddings and returns:
      - acc@k for each k in `topk`
      - precision_macro, recall_macro, f1_macro, confusion_matrix
      - per_category (optional; includes acc@1 and acc@5 if requested)
      - (extra) topk_idx/topk_sim for convenience (max over requested k)
    """
    dev = device or _device()
    gcms_encoder.to(dev).eval()
    sensor_encoder.to(dev).eval()

    # to tensors
    if not torch.is_tensor(gcms_data):      gcms_data = torch.tensor(gcms_data)
    if not torch.is_tensor(sensor_data):    sensor_data = torch.tensor(sensor_data)
    if not torch.is_tensor(sensor_labels):  sensor_labels = torch.tensor(sensor_labels, dtype=torch.long)

    gcms_data = _maybe_to_device(gcms_data, dev, dtype)
    sensor_labels = _maybe_to_device(sensor_labels, dev)
    lengths = _maybe_to_device(lengths, dev)

    # encode GCMS in one go (usually small)
    with torch.no_grad():
        zg = gcms_encoder.forward_features(gcms_data) if hasattr(gcms_encoder, "forward_features") else gcms_encoder(gcms_data)
        if l2_normalize:
            zg = F.normalize(zg, dim=1)

    # encode sensor (optionally in chunks)
    with torch.no_grad():
        zs_list = []
        if batch_size is None:
            sd = _maybe_to_device(sensor_data, dev, dtype)
            len_batch = lengths
            z = sensor_encoder.forward_features(sd, lengths=len_batch) if hasattr(sensor_encoder, "forward_features") else sensor_encoder(sd)
            zs_list.append(z)
        else:
            N = sensor_data.size(0)
            for i in range(0, N, batch_size):
                sd = _maybe_to_device(sensor_data[i:i+batch_size], dev, dtype)
                len_batch = None if lengths is None else lengths[i:i+batch_size]
                z = sensor_encoder.forward_features(sd, lengths=len_batch) if hasattr(sensor_encoder, "forward_features") else sensor_encoder(sd)
                zs_list.append(z)

        zs = torch.cat(zs_list, dim=0)
        if l2_normalize:
            zs = F.normalize(zs, dim=1)

        # cosine similarity matrix (N_s, N_g)
        sim = zs @ zg.T

        # predictions
        top1 = sim.argmax(dim=1)  # (N_s,)
        y = sensor_labels
        total = y.size(0)

        # acc@k (generic)
        results: Dict[str, Union[float, np.ndarray, Dict]] = {}
        need_topk = any(k > 1 for k in topk)
        max_k = min(max(topk), sim.shape[1])
        # precompute a single topk up to max requested k for convenience exports
        topk_val_all, topk_idx_all = torch.topk(sim, k=max_k, dim=1)

        # compute acc@k exactly for requested k (not just max_k)
        for k in topk:
            kk = min(k, sim.shape[1])
            idx_k = topk_idx_all[:, :kk]
            acc_k = (idx_k == y.unsqueeze(1)).any(dim=1).float().mean().item() * 100.0
            results[f"acc@{k}"] = acc_k

    # macro metrics (align with `evaluate`)
    y_np = y.cpu().numpy()
    p_np = top1.cpu().numpy()
    if total > 0:
        results["precision_macro"] = precision_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["recall_macro"]    = recall_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["f1_macro"]        = f1_score(y_np, p_np, average="macro", zero_division=0) * 100
        results["confusion_matrix"] = confusion_matrix(y_np, p_np)

    # convenience exports (optional; remove if you want an exact match)
    results["topk_idx"] = topk_idx_all[:, :max_k].cpu().numpy()
    results["topk_sim"] = topk_val_all[:, :max_k].cpu().numpy()

    # --- per-category (by TRUE category of sensor_labels) ---
    if (ingredient_to_category is not None) and (class_names is not None) and (total > 0):
        class_to_cat, missing = _build_class_to_category(class_names, ingredient_to_category)
        if logger and missing:
            logger.warning(f"[contrastive] {len(missing)} classes missing in ingredient_to_category; counted as 'UNKNOWN'.")

        true_cat = np.array([class_to_cat[int(c)] for c in y_np], dtype=object)
        correct1 = (p_np == y_np)

        # prepare hits@5 if requested (to mirror `evaluate`'s shape/keys)
        per_cat: Dict[str, Dict[str, float]] = {}
        include_acc5 = (need_topk and 5 in topk)
        hits5 = None
        if include_acc5:
            kk = min(5, sim.shape[1])
            idx5 = topk_idx_all[:, :kk].cpu().numpy()
            hits5 = (idx5 == y_np[:, None]).any(axis=1)

        for cat in np.unique(true_cat):
            mask = (true_cat == cat)
            n = int(mask.sum())
            row = {
                "n": n,
                "acc@1": float(correct1[mask].mean()*100.0) if n > 0 else 0.0,
            }
            if include_acc5:
                row["acc@5"] = float(hits5[mask].mean()*100.0) if n > 0 else 0.0
            per_cat[str(cat)] = row

        results["per_category"] = per_cat  # {category: {"n": int, "acc@1": %, "acc@5": %?}}

    # logging mirrors `evaluate`
    if logger:
        msg = " | ".join([f"acc@{k}: {results[f'acc@{k}']:.2f}%" for k in topk])
        logger.info(f"✅ Contrastive — {msg}")
        if "per_category" in results:
            logger.info("Per-category acc:")
            for cat, row in results["per_category"].items():
                extras = f", acc@5={row['acc@5']:.2f}%" if ("acc@5" in row) else ""
                logger.info(f"  - {cat}: n={row['n']}, acc@1={row['acc@1']:.2f}%{extras}")
    else:
        printable = {k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in results.items() if "matrix" not in k and not isinstance(v, dict)}
        print(printable)

    return results
