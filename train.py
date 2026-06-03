import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from safetensors.torch import load_file as safe_load_file
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn

# ===== AMP 兼容写法：兼容新旧 PyTorch =====
try:
    from torch.amp import GradScaler, autocast
    AMP_NEW_API = True
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    AMP_NEW_API = False

from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from timm.data import resolve_model_data_config


@dataclass
class ModelConfig:
    name: str
    family: str
    model_id: str
    local_ckpt: str
    batch_size: int
    grad_accum_steps: int
    lr: float
    head_lr_mult: float
    weight_decay: float
    epochs: int
    patience: int


def make_grad_scaler(device: torch.device, enabled: Optional[bool] = None):
    if enabled is None:
        enabled = (device.type == "cuda")

    if AMP_NEW_API:
        try:
            return GradScaler("cuda", enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)
    else:
        return GradScaler(enabled=enabled)


def amp_autocast(device: torch.device):
    if AMP_NEW_API:
        return autocast(device_type="cuda", enabled=(device.type == "cuda"))
    else:
        return autocast(enabled=(device.type == "cuda"))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv_rows(rows: List[List[Any]], path: Union[str, Path]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def check_dataset_structure(data_root: str) -> None:
    required = [
        Path(data_root) / "train",
        Path(data_root) / "val",
        Path(data_root) / "test",
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError("Missing folder: {}".format(p))

    split_classes = {}
    for split in required:
        subdirs = sorted([x.name for x in split.iterdir() if x.is_dir()])
        if len(subdirs) == 0:
            raise RuntimeError("No class folders found in: {}".format(split))
        split_classes[split.name] = subdirs

    train_classes = split_classes["train"]
    for split_name in ["val", "test"]:
        if split_classes[split_name] != train_classes:
            raise RuntimeError(
                "Class folders mismatch between train and {}.\ntrain: {}\n{}: {}".format(
                    split_name, train_classes, split_name, split_classes[split_name]
                )
            )


def get_num_classes_and_names(data_root: str) -> Tuple[int, List[str]]:
    train_dir = os.path.join(data_root, "train")
    ds = datasets.ImageFolder(train_dir)
    return len(ds.classes), ds.classes


def parse_checkpoint(ckpt_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    ckpt_path = str(ckpt_path)
    suffix = Path(ckpt_path).suffix.lower()

    if suffix == ".safetensors":
        state_dict = safe_load_file(ckpt_path)
        return state_dict

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state_dict", "params"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

    if isinstance(obj, dict):
        return obj

    raise RuntimeError("Unsupported checkpoint format: {}".format(ckpt_path))


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        cleaned[new_k] = v
    return cleaned


def load_local_pretrained(model: nn.Module, ckpt_path: Union[str, Path]) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError("Local checkpoint not found: {}".format(ckpt_path))

    raw_state = parse_checkpoint(ckpt_path)
    raw_state = normalize_state_dict_keys(raw_state)

    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for k, v in raw_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            skipped_keys.append(k)

    incompatible = model.load_state_dict(filtered_state, strict=False)

    print("[INFO] Loaded local checkpoint: {}".format(ckpt_path))
    print("[INFO] Matched params      : {}".format(len(filtered_state)))
    print("[INFO] Skipped params      : {}".format(len(skipped_keys)))
    if len(incompatible.missing_keys) > 0:
        print("[INFO] Missing keys       : {}".format(len(incompatible.missing_keys)))
    if len(incompatible.unexpected_keys) > 0:
        print("[INFO] Unexpected keys    : {}".format(len(incompatible.unexpected_keys)))


def build_model(cfg: ModelConfig, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg.model_id,
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )
    load_local_pretrained(model, cfg.local_ckpt)
    return model


def get_classifier_param_ids(model: nn.Module) -> set:
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
        if isinstance(head, nn.Module):
            return set(id(p) for p in head.parameters())
    return set()


def build_optimizer(model: nn.Module, cfg: ModelConfig) -> AdamW:
    head_param_ids = get_classifier_param_ids(model)

    decay_backbone = []
    no_decay_backbone = []
    decay_head = []
    no_decay_head = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_head = id(p) in head_param_ids
        no_decay = (
            p.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "bn" in name.lower()
        )

        if is_head and no_decay:
            no_decay_head.append(p)
        elif is_head and not no_decay:
            decay_head.append(p)
        elif (not is_head) and no_decay:
            no_decay_backbone.append(p)
        else:
            decay_backbone.append(p)

    param_groups = [
        {"params": decay_backbone, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": no_decay_backbone, "lr": cfg.lr, "weight_decay": 0.0},
        {"params": decay_head, "lr": cfg.lr * cfg.head_lr_mult, "weight_decay": cfg.weight_decay},
        {"params": no_decay_head, "lr": cfg.lr * cfg.head_lr_mult, "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_transforms_for_model(cfg: ModelConfig):
    temp_model = timm.create_model(cfg.model_id, pretrained=False, num_classes=1000)
    data_cfg = resolve_model_data_config(temp_model)
    input_size = data_cfg["input_size"][-1]
    mean = data_cfg["mean"]
    std = data_cfg["std"]
    del temp_model

    resize_size = int(round(input_size * 1.14))

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size,
                scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf, input_size


def make_datasets(data_root: str, train_tf, eval_tf):
    train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), transform=eval_tf)
    return train_ds, val_ds, test_ds


def build_weighted_sampler(train_ds: datasets.ImageFolder) -> WeightedRandomSampler:
    targets = np.array(train_ds.targets)
    class_sample_count = np.bincount(targets)
    class_sample_count = np.maximum(class_sample_count, 1)
    weights = 1.0 / class_sample_count
    sample_weights = weights[targets]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def build_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool = True,
):
    train_sampler = build_weighted_sampler(train_ds) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader


def forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    return model(images)


def compute_auc_ovr_macro(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Optional[float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        print("[WARN] AUC skipped: empty y_true.")
        return None

    num_classes = len(class_names)

    if y_prob.ndim != 2:
        print("[WARN] AUC skipped: y_prob.ndim != 2, got {}".format(y_prob.ndim))
        return None

    # ===== 二分类：直接计算正类 AUC =====
    if num_classes == 2:
        if y_prob.shape[1] == 2:
            pos_prob = y_prob[:, 1]
        elif y_prob.shape[1] == 1:
            pos_prob = y_prob[:, 0]
        else:
            print(
                "[WARN] AUC skipped: probability dimension mismatch for binary task, "
                "y_prob.shape={}".format(y_prob.shape)
            )
            return None

        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            print("[WARN] AUC skipped: binary y_true has only one class.")
            return None

        try:
            return float(roc_auc_score(y_true, pos_prob))
        except Exception as e:
            print("[WARN] Binary AUC failed: {}".format(e))
            return None

    # ===== 多分类：One-vs-Rest Macro AUC =====
    if y_prob.shape[1] != num_classes:
        print(
            "[WARN] AUC skipped: probability dimension mismatch, "
            "y_prob.shape[1]={}, num_classes={}".format(y_prob.shape[1], num_classes)
        )
        return None

    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    valid_class_indices = []
    for c in range(num_classes):
        positives = int(y_true_bin[:, c].sum())
        negatives = int(len(y_true_bin) - positives)
        if positives > 0 and negatives > 0:
            valid_class_indices.append(c)

    if len(valid_class_indices) == 0:
        print("[WARN] AUC skipped: no valid one-vs-rest class has both positive and negative samples.")
        return None

    auc_list = []
    auc_class_names = []
    for c in valid_class_indices:
        try:
            auc_c = roc_auc_score(y_true_bin[:, c], y_prob[:, c])
            auc_list.append(float(auc_c))
            auc_class_names.append(class_names[c])
        except Exception as e:
            print("[WARN] AUC failed for class '{}': {}".format(class_names[c], e))

    if len(auc_list) == 0:
        print("[WARN] AUC skipped: all class-wise AUC computations failed.")
        return None

    if len(auc_list) < num_classes:
        missing = [class_names[i] for i in range(num_classes) if class_names[i] not in auc_class_names]
        print(
            "[WARN] AUC computed on present/valid classes only. "
            "Used classes: {} | Skipped classes: {}".format(auc_class_names, missing)
        )

    return float(np.mean(auc_list))


def compute_metrics(y_true, y_pred, y_prob, class_names: List[str]) -> Dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None:
        metrics["auc_ovr_macro"] = compute_auc_ovr_macro(y_true, y_prob, class_names)
    else:
        metrics["auc_ovr_macro"] = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    criterion,
    scaler,
    device: torch.device,
    train: bool,
    grad_accum_steps: int = 1,
):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, ncols=120, leave=False)
    for step, batch in enumerate(pbar, start=1):
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with amp_autocast(device):
                logits = forward_logits(model, images)
                loss = criterion(logits, targets)
                if train:
                    loss = loss / grad_accum_steps

            if train:
                scaler.scale(loss).backward()

                if step % grad_accum_steps == 0 or step == len(loader):
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()

                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None and new_scale >= old_scale:
                        scheduler.step()

        batch_loss = float(loss.detach().item()) * (grad_accum_steps if train else 1.0)
        running_loss += batch_loss * images.size(0)

        probs = torch.softmax(logits.detach(), dim=1)
        preds = probs.argmax(dim=1)

        all_true.extend(targets.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())
        all_prob.extend(probs.detach().cpu().numpy().tolist())

        current_acc = accuracy_score(all_true, all_pred) if len(all_true) else 0.0
        pbar.set_description(
            "{} loss={:.4f} acc={:.4f}".format("train" if train else "eval ", batch_loss, current_acc)
        )

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(all_true), np.array(all_pred), np.array(all_prob)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    class_names: List[str],
):
    loss, y_true, y_pred, y_prob = run_one_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        scheduler=None,
        criterion=criterion,
        scaler=make_grad_scaler(device, enabled=False),
        device=device,
        train=False,
        grad_accum_steps=1,
    )
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    metrics["loss"] = float(loss)
    return metrics


def save_confusion_matrix_csv(
    conf_mat: List[List[int]],
    class_names: List[str],
    path: Union[str, Path]
) -> None:
    rows = [["label"] + class_names]
    for name, row in zip(class_names, conf_mat):
        rows.append([name] + row)
    save_csv_rows(rows, path)


def save_report_csv(report: Dict[str, Any], path: Union[str, Path]) -> None:
    rows = [["label", "precision", "recall", "f1-score", "support"]]
    for key, value in report.items():
        if isinstance(value, dict):
            rows.append([
                key,
                value.get("precision"),
                value.get("recall"),
                value.get("f1-score"),
                value.get("support"),
            ])
    save_csv_rows(rows, path)


def train_single_model(
    cfg: ModelConfig,
    data_root: str,
    output_root: str,
    device: torch.device,
    num_workers: int,
    seed: int,
):
    model_out = Path(output_root) / cfg.name
    ensure_dir(model_out)
    save_json(asdict(cfg), model_out / "model_config.json")

    num_classes, class_names = get_num_classes_and_names(data_root)

    train_tf, eval_tf, input_size = build_transforms_for_model(cfg)
    train_ds, val_ds, test_ds = make_datasets(data_root, train_tf, eval_tf)
    train_loader, val_loader, test_loader = build_loaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        use_weighted_sampler=True,
    )

    model = build_model(cfg, num_classes).to(device)
    optimizer = build_optimizer(model, cfg)
    total_steps = int(math.ceil(len(train_loader) / float(cfg.grad_accum_steps)) * cfg.epochs)
    scheduler = build_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)
    scaler = make_grad_scaler(device, enabled=(device.type == "cuda"))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    bincounts = np.bincount(train_ds.targets)
    class_count = dict((class_names[i], int(c)) for i, c in enumerate(bincounts))
    save_json(class_count, model_out / "train_class_count.json")

    history = []
    best_score = -1.0
    best_epoch = -1
    best_ckpt = model_out / "best.pth"
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_true, train_pred, train_prob = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            train=True,
            grad_accum_steps=cfg.grad_accum_steps,
        )
        train_metrics = compute_metrics(train_true, train_pred, train_prob, class_names)
        train_metrics["loss"] = float(train_loss)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
        )

        epoch_info = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[2]["lr"],
        }
        history.append(epoch_info)
        save_json(history, model_out / "history.json")

        score = val_metrics["accuracy"]
        print(
            "[{}] epoch {:02d} | train loss {:.4f} acc {:.4f} macro_f1 {:.4f} | "
            "val loss {:.4f} acc {:.4f} macro_f1 {:.4f}".format(
                cfg.name,
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["macro_f1"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
            )
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_accuracy": best_score,
                    "class_names": class_names,
                    "model_cfg": asdict(cfg),
                    "input_size": input_size,
                    "seed": seed,
                },
                best_ckpt,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("[{}] Early stopping at epoch {}, best epoch = {}".format(cfg.name, epoch, best_epoch))
                break

    print("[{}] loading best checkpoint from epoch {}".format(cfg.name, best_epoch))
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_best = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
    )
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
    )

    save_json(val_best, model_out / "best_val_metrics.json")
    save_json(test_metrics, model_out / "test_metrics.json")
    save_report_csv(test_metrics["classification_report"], model_out / "test_classification_report.csv")
    save_confusion_matrix_csv(test_metrics["confusion_matrix"], class_names, model_out / "test_confusion_matrix.csv")

    summary = {
        "model_name": cfg.name,
        "model_id": cfg.model_id,
        "local_ckpt": cfg.local_ckpt,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_score,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_auc_ovr_macro": test_metrics["auc_ovr_macro"],
        "output_dir": str(model_out),
    }
    save_json(summary, model_out / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="dataset root with train/val/test")
    parser.add_argument("--output_root", type=str, default="./runs_transformer_3models")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--model_id",
        type=str,
        default="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        help="timm model architecture name",
    )
    parser.add_argument(
        "--swin_ckpt",
        type=str,
        required=True,
        help="local checkpoint path, e.g. /home/c/weing/fenbian/local_models/swinv2_large/model.safetensors",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)

    args = parser.parse_args()

    check_dataset_structure(args.data_root)
    ensure_dir(args.output_root)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print("Device      : {}".format(device))
    print("Data root   : {}".format(args.data_root))
    print("Output root : {}".format(args.output_root))
    print("Model       : swinv2_large")
    print("Model ID    : {}".format(args.model_id))
    print("Checkpoint  : {}".format(args.swin_ckpt))
    print("=" * 100)

    cfg = ModelConfig(
        name="swinv2_large",
        family="timm",
        model_id=args.model_id,
        local_ckpt=args.swin_ckpt,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        head_lr_mult=args.head_lr_mult,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )

    result = train_single_model(
        cfg=cfg,
        data_root=args.data_root,
        output_root=args.output_root,
        device=device,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    save_json([result], Path(args.output_root) / "all_results.json")
    save_json([result], Path(args.output_root) / "all_results_sorted.json")

    rows = [[
        "rank", "model_name", "model_id", "local_ckpt", "best_epoch",
        "best_val_accuracy", "test_accuracy", "test_macro_f1",
        "test_weighted_f1", "test_auc_ovr_macro", "output_dir"
    ]]
    rows.append([
        1,
        result["model_name"],
        result["model_id"],
        result["local_ckpt"],
        result["best_epoch"],
        result["best_val_accuracy"],
        result["test_accuracy"],
        result["test_macro_f1"],
        result["test_weighted_f1"],
        result["test_auc_ovr_macro"],
        result["output_dir"],
    ])
    save_csv_rows(rows, Path(args.output_root) / "leaderboard.csv")

    print("\nFinal result:")
    print(
        "1. {}: test_acc={:.4f}, test_macro_f1={:.4f}, test_weighted_f1={:.4f}, test_auc={}".format(
            result["model_name"],
            result["test_accuracy"],
            result["test_macro_f1"],
            result["test_weighted_f1"],
            result["test_auc_ovr_macro"],
        )
    )


if __name__ == "__main__":
    main()

# 运行示例：
# python train_transformer_3models.py --data_root /home/a/vv/DDH/DDH_split_721 --output_root ./runs_transformer_DDH_2 --num_workers 8 --swin_ckpt /home/a/vv/DDH/local_models/swinv2_large/model.safetensors

# 显存不够时：
# python train_transformer_3models.py --data_root /home/a/vv/DDH/DDH_grouped/DDH_4cls --output_root ./runs_transformer_DDH_4class --num_workers 8 --swin_ckpt /home/a/vv/DDH/local_models/swinv2_large/model.safetensors --batch_size 4 --grad_accum_steps 2
