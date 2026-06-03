import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import timm
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from torchvision import transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULT_CONFIGS = {
    2: {
        "test_root": r"/home/a/vv/DDH/DDH_grouped/DDH_2cls/test",
        "train_root": r"/home/a/vv/DDH/DDH_grouped/DDH_2cls/train",
        "model_path": r"/home/a/vv/DDH/runs_transformer_DDH_2class/swinv2_large/best.pth",
        "output_dir": r"/home/a/vv/DDH/ddh_2cls_cam",
        "class_names": ["I", "Others"],
        "default_target_class": "Others",
    },
    3: {
        "test_root": r"/home/a/vv/DDH/DDH_grouped/DDH_3cls/test",
        "train_root": r"/home/a/vv/DDH/DDH_grouped/DDH_3cls/train",
        "model_path": r"/home/a/vv/DDH/runs_transformer_DDH_3class/swinv2_large/best.pth",
        "output_dir": r"/home/a/vv/DDH/ddh_3cls_cam",
        "class_names": ["D_III_IV", "I", "IIabc"],
        "default_target_class": "IIabc",
    },
    4: {
        "test_root": r"/home/a/vv/DDH/DDH_grouped/DDH_4cls/test",
        "train_root": r"/home/a/vv/DDH/DDH_grouped/DDH_4cls/train",
        "model_path": r"/home/a/vv/DDH/runs_transformer_DDH_4class/swinv2_large/best.pth",
        "output_dir": r"/home/a/vv/DDH/ddh_4cls_cam",
        "class_names": ["D_III_IV", "I", "IIab", "IIc"],
        "default_target_class": "IIc",
    },
}


def is_image_file(path):
    return Path(path).suffix.lower() in IMG_EXTENSIONS


def collect_images(test_root):
    image_paths = []
    test_root = Path(test_root)

    for root, _, files in os.walk(test_root):
        for file in files:
            file_path = Path(root) / file
            if is_image_file(file_path):
                image_paths.append(file_path)

    return sorted(image_paths)


def clean_state_dict(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        for prefix in ["module.", "_orig_mod.", "model."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_state_dict[k] = v

    return new_state_dict


def torch_load_checkpoint(model_path):
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location="cpu")

    return ckpt


def load_checkpoint(model_path):
    ckpt = torch_load_checkpoint(model_path)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "net" in ckpt:
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        raise TypeError("当前脚本只支持 state_dict 或包含 state_dict 的 checkpoint。")

    state_dict = clean_state_dict(state_dict)
    return state_dict, ckpt


def infer_num_classes(state_dict):
    candidate_keys = [
        "head.fc.weight",
        "head.weight",
        "classifier.weight",
        "fc.weight",
    ]

    for key in candidate_keys:
        if key in state_dict:
            return state_dict[key].shape[0]

    for key, value in state_dict.items():
        if key.endswith("weight") and value.ndim == 2:
            return value.shape[0]

    raise ValueError("无法从 pth 自动推断类别数，请手动指定 --num_classes。")


def get_class_names(ckpt, test_root, train_root, default_class_names):
    if isinstance(ckpt, dict) and "class_to_idx" in ckpt:
        class_to_idx = ckpt["class_to_idx"]
        class_names = [None] * len(class_to_idx)

        for name, idx in class_to_idx.items():
            class_names[idx] = name

        return class_names

    if train_root is not None and Path(train_root).exists():
        class_names = sorted([
            p.name for p in Path(train_root).iterdir()
            if p.is_dir()
        ])
        if len(class_names) > 0:
            return class_names

    if test_root is not None and Path(test_root).exists():
        class_names = sorted([
            p.name for p in Path(test_root).iterdir()
            if p.is_dir()
        ])
        if len(class_names) > 0:
            return class_names

    return default_class_names


def get_true_class(image_path, test_root):
    image_path = Path(image_path)
    test_root = Path(test_root)

    try:
        rel_path = image_path.relative_to(test_root)
        if len(rel_path.parts) >= 2:
            return rel_path.parts[0]
    except ValueError:
        pass

    return ""


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def load_image_tensor(image_path, img_size, device):
    pil_img = Image.open(image_path).convert("RGB")
    original_size = pil_img.size  # (W, H)

    transform = build_transform(img_size)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    resized_rgb = np.asarray(pil_img.resize((img_size, img_size)), dtype=np.float32) / 255.0

    return pil_img, resized_rgb, input_tensor, original_size


def build_model(model_name, num_classes, state_dict, device):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        print("[Warning] Missing keys:")
        for k in missing_keys:
            print("  ", k)

    if len(unexpected_keys) > 0:
        print("[Warning] Unexpected keys:")
        for k in unexpected_keys:
            print("  ", k)

    model = model.to(device)
    model.eval()
    return model


def get_module_by_name(model, module_name):
    named_modules = dict(model.named_modules())
    if module_name not in named_modules:
        available = [name for name in named_modules.keys() if name]
        tail = "\n".join(available[-80:])
        raise ValueError(
            f"找不到 target_layer={module_name}。\n"
            f"可以先运行 --list_layers 查看可用层名。\n"
            f"部分可用层名如下：\n{tail}"
        )
    return named_modules[module_name]


def guess_target_layer_name(model):
    """
    自动选择一个较靠后的归一化层作为 Grad-CAM 的 target layer。

    对 timm Swin/SwinV2，通常会选到类似：
        layers.3.blocks.1.norm2
        stages.3.blocks.1.norm2
    如果模型结构不同，会退化为最后一个 norm 层。
    """
    named_modules = list(model.named_modules())
    module_names = [name for name, _ in named_modules if name]

    priority_keywords = [
        ".blocks.",
        "norm2",
        "norm1",
        "norm",
    ]

    # 优先找最后 stage / layer 中最后 block 的 norm2。
    candidates = []
    for name, module in named_modules:
        if not name:
            continue
        lower = name.lower()
        cls_name = module.__class__.__name__.lower()
        if "norm2" in lower and ("layers." in lower or "stages." in lower) and ".blocks." in lower:
            candidates.append(name)
    if len(candidates) > 0:
        return candidates[-1]

    # 再找最后一个 block 内 norm。
    candidates = []
    for name, module in named_modules:
        if not name:
            continue
        lower = name.lower()
        cls_name = module.__class__.__name__.lower()
        if ".blocks." in lower and ("norm" in lower or "norm" in cls_name):
            candidates.append(name)
    if len(candidates) > 0:
        return candidates[-1]

    # 最后退化为最后一个 LayerNorm / BatchNorm / GroupNorm 类模块。
    candidates = []
    for name, module in named_modules:
        cls_name = module.__class__.__name__.lower()
        if "norm" in name.lower() or "norm" in cls_name:
            candidates.append(name)
    if len(candidates) > 0:
        return candidates[-1]

    # 实在没有就返回最后一个非空模块名。
    if len(module_names) == 0:
        raise RuntimeError("模型中没有可用模块，无法选择 Grad-CAM target layer。")
    return module_names[-1]


def list_candidate_layers(model):
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        lower = name.lower()
        cls_name = module.__class__.__name__
        if (
            "norm" in lower
            or "blocks" in lower
            or "layer" in lower
            or "stage" in lower
            or "conv" in lower
        ):
            rows.append((name, cls_name))
    return rows


def tensor_to_cam_feature(tensor):
    """
    将 hook 得到的 activation / gradient 转成 B,C,H,W。

    支持常见输出：
    1. B,C,H,W
    2. B,H,W,C  Swin 常见 channels-last
    3. B,L,C    Transformer token 序列，要求 L 可以开平方
    """
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]

    if tensor.ndim == 4:
        # B,C,H,W
        if tensor.shape[1] <= tensor.shape[-1] and tensor.shape[1] in [1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]:
            return tensor
        # B,H,W,C -> B,C,H,W
        return tensor.permute(0, 3, 1, 2)

    if tensor.ndim == 3:
        b, l, c = tensor.shape
        h = int(np.sqrt(l))
        w = h
        if h * w != l:
            raise RuntimeError(
                f"当前 target layer 输出为 B,L,C，但 L={l} 不能开平方成二维特征图。"
                f"请尝试指定更合适的层，例如 --target_layer layers.3.blocks.1.norm2"
            )
        return tensor.transpose(1, 2).reshape(b, c, h, w)

    raise RuntimeError(
        f"当前 target layer 输出维度为 {tuple(tensor.shape)}，无法自动转换为 CAM 特征图。"
    )


class ManualGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, input_tensor, target_index):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        score = logits[:, target_index].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "没有捕获到 activation 或 gradient。请尝试更换 --target_layer。"
            )

        activations = tensor_to_cam_feature(self.activations)
        gradients = tensor_to_cam_feature(self.gradients)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * activations).sum(dim=1, keepdim=False)
        cam_map = F.relu(cam_map)

        cam_map = cam_map[0]
        cam_map = cam_map - cam_map.min()
        cam_max = cam_map.max()
        if cam_max > 0:
            cam_map = cam_map / cam_max

        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_conf = float(probs[0, pred_idx].detach().cpu().item())
        target_score = float(probs[0, target_index].detach().cpu().item())
        probs_list = probs[0].detach().cpu().tolist()

        return cam_map.detach().cpu().numpy(), pred_idx, pred_conf, target_score, probs_list


def resize_cam_to_image(cam_map, img_size):
    cam_tensor = torch.from_numpy(cam_map).float().unsqueeze(0).unsqueeze(0)
    cam_tensor = F.interpolate(cam_tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    cam_resized = cam_tensor.squeeze().numpy()
    cam_resized = cam_resized - cam_resized.min()
    cam_max = cam_resized.max()
    if cam_max > 0:
        cam_resized = cam_resized / cam_max
    return cam_resized


def make_heatmap(cam_resized, colormap_name="jet"):
    cmap = plt.get_cmap(colormap_name)
    heatmap = cmap(cam_resized)[:, :, :3]
    return heatmap.astype(np.float32)


def overlay_heatmap(rgb_image, heatmap, alpha=0.45):
    overlay = (1.0 - alpha) * rgb_image + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def save_cam_images(
    rgb_image,
    cam_resized,
    heatmap,
    overlay,
    output_stem,
    title,
):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    original_path = output_stem.with_suffix(".original.png")
    heatmap_path = output_stem.with_suffix(".heatmap.png")
    overlay_path = output_stem.with_suffix(".overlay.png")
    panel_path = output_stem.with_suffix(".panel.png")

    plt.imsave(original_path, rgb_image)
    plt.imsave(heatmap_path, heatmap)
    plt.imsave(overlay_path, overlay)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "panel_path": str(panel_path),
    }


def safe_name(text):
    text = str(text)
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    return text


def should_keep_image(true_class, pred_class, args):
    if args.only_true_class is not None and true_class != args.only_true_class:
        return False
    if args.only_pred_class is not None and pred_class != args.only_pred_class:
        return False
    if args.only_correct and true_class != "" and true_class != pred_class:
        return False
    if args.only_wrong and true_class != "" and true_class == pred_class:
        return False
    return True


def limit_images_by_class(image_paths, test_root, max_per_class):
    if max_per_class is None or max_per_class <= 0:
        return image_paths

    kept = []
    counter = {}
    for path in image_paths:
        cls = get_true_class(path, test_root)
        counter.setdefault(cls, 0)
        if counter[cls] < max_per_class:
            kept.append(path)
            counter[cls] += 1
    return kept


def apply_threshold_prediction(probs, class_names, target_class, threshold):
    class_names = list(class_names)
    probs = np.asarray(probs, dtype=np.float64)

    if target_class is None:
        raise ValueError("使用阈值预测时必须指定 --target_class。")
    if target_class not in class_names:
        raise ValueError(f"target_class={target_class} 不在类别列表中: {class_names}")

    target_idx = class_names.index(target_class)

    if len(class_names) == 2:
        neg_idx = [i for i in range(len(class_names)) if i != target_idx][0]
        if probs[target_idx] >= threshold:
            return target_idx
        return neg_idx

    # 多分类：如果目标类概率超过阈值，则预测目标类；否则在非目标类中选最大概率。
    non_target_indices = [i for i in range(len(class_names)) if i != target_idx]
    if probs[target_idx] >= threshold:
        return target_idx
    return non_target_indices[int(np.argmax(probs[non_target_indices]))]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--test_root", type=str, default=None)
    parser.add_argument("--train_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        help="timm 模型名称，需要和训练时保持一致。"
    )
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="模型输出类别顺序，例如三分类: --class_names D_III_IV I IIabc"
    )

    parser.add_argument(
        "--target_layer",
        type=str,
        default="auto",
        help="Grad-CAM 目标层名。默认 auto。可用 --list_layers 查看。"
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="只打印候选层名，不生成 CAM。"
    )

    parser.add_argument(
        "--cam_target",
        type=str,
        default="both",
        choices=["pred", "true", "class", "both"],
        help=(
            "CAM 目标类别。pred=预测类别；true=真实类别；class=指定 --target_class；"
            "both=同时保存 pred 和 target_class。"
        )
    )
    parser.add_argument(
        "--target_class",
        type=str,
        default=None,
        help="指定 CAM 目标类。二分类默认 Others，三分类默认 IIabc。"
    )

    parser.add_argument(
        "--prediction_rule",
        type=str,
        default="argmax",
        choices=["argmax", "target_threshold"],
        help="用于记录 pred_class 的规则。CAM 本身仍可指定目标类别。"
    )
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=None,
        help="prediction_rule=target_threshold 时使用，例如三分类 IIabc 阈值 0.086。"
    )

    parser.add_argument("--max_images", type=int, default=60, help="最多生成多少张图片的 CAM。<=0 表示全部。")
    parser.add_argument("--max_per_class", type=int, default=10, help="每个真实类别最多取多少张。<=0 表示不限制。")
    parser.add_argument("--only_true_class", type=str, default=None, help="只处理指定真实类别。")
    parser.add_argument("--only_pred_class", type=str, default=None, help="只处理指定预测类别。")
    parser.add_argument("--only_correct", action="store_true", help="只处理预测正确样本。")
    parser.add_argument("--only_wrong", action="store_true", help="只处理预测错误样本。")

    parser.add_argument("--alpha", type=float, default=0.45, help="热图叠加透明度。")
    parser.add_argument("--colormap", type=str, default="jet", help="matplotlib colormap，例如 jet、turbo、magma。")
    parser.add_argument("--device", type=str, default=None, help="默认自动选择 cuda 或 cpu。")

    args = parser.parse_args()

    cfg = DEFAULT_CONFIGS[args.num_classes]

    test_root = Path(args.test_root if args.test_root is not None else cfg["test_root"])
    train_root = Path(args.train_root if args.train_root is not None else cfg["train_root"])
    model_path = Path(args.model_path if args.model_path is not None else cfg["model_path"])
    output_dir = Path(args.output_dir if args.output_dir is not None else cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.target_class is None:
        target_class = cfg.get("default_target_class", None)
    else:
        target_class = args.target_class

    if not test_root.exists():
        raise FileNotFoundError(f"测试集路径不存在: {test_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, ckpt = load_checkpoint(model_path)
    pth_num_classes = infer_num_classes(state_dict)
    if pth_num_classes != args.num_classes:
        raise ValueError(
            f"当前 pth 的分类头类别数为 {pth_num_classes}，"
            f"但你设置的是 num_classes={args.num_classes}。"
        )

    if args.class_names is not None:
        class_names = args.class_names
    else:
        class_names = get_class_names(
            ckpt=ckpt,
            test_root=test_root,
            train_root=train_root,
            default_class_names=cfg["class_names"],
        )

    if len(class_names) != args.num_classes:
        raise ValueError(
            f"类别名称数量和模型类别数不一致: len(class_names)={len(class_names)}, "
            f"num_classes={args.num_classes}。请手动指定 --class_names。"
        )

    if target_class is not None and target_class not in class_names:
        raise ValueError(f"target_class={target_class} 不在模型输出类别列表中: {class_names}")

    print("=" * 100)
    print(f"Device                  : {device}")
    print(f"Num classes             : {args.num_classes}")
    print(f"Model output class names: {class_names}")
    print(f"Default CAM target class: {target_class}")
    print(f"Model name              : {args.model_name}")
    print(f"Model path              : {model_path}")
    print(f"Test root               : {test_root}")
    print(f"Output dir              : {output_dir}")
    print(f"CAM target mode         : {args.cam_target}")
    print(f"Prediction rule         : {args.prediction_rule}")
    print("=" * 100)

    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        state_dict=state_dict,
        device=device,
    )

    if args.list_layers:
        rows = list_candidate_layers(model)
        print("可选 target layer 候选如下：")
        for name, cls_name in rows:
            print(f"{name:80s} {cls_name}")
        return

    if args.target_layer == "auto":
        target_layer_name = guess_target_layer_name(model)
    else:
        target_layer_name = args.target_layer

    target_layer = get_module_by_name(model, target_layer_name)
    print(f"Grad-CAM target layer   : {target_layer_name} ({target_layer.__class__.__name__})")

    if args.prediction_rule == "target_threshold":
        if args.positive_threshold is None:
            raise ValueError("prediction_rule=target_threshold 时必须设置 --positive_threshold。")
        if target_class is None:
            raise ValueError("prediction_rule=target_threshold 时必须设置 --target_class。")

    image_paths = collect_images(test_root)
    if len(image_paths) == 0:
        raise RuntimeError(f"没有在测试集目录中找到图片: {test_root}")

    image_paths = limit_images_by_class(
        image_paths=image_paths,
        test_root=test_root,
        max_per_class=args.max_per_class,
    )

    if args.max_images is not None and args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    cam_engine = ManualGradCAM(model=model, target_layer=target_layer)
    summary_rows = []

    try:
        for image_path in tqdm(image_paths, desc="Generating DDH CAM"):
            rel_path = image_path.relative_to(test_root).as_posix()
            true_class = get_true_class(image_path, test_root)

            pil_img, rgb_image, input_tensor, _ = load_image_tensor(
                image_path=image_path,
                img_size=args.img_size,
                device=device,
            )

            # 先跑一次 pred CAM 或 class CAM 时会获得预测概率。
            # 为了根据 pred_class 筛选，这里先用 no_grad 做一次普通前向。
            with torch.no_grad():
                logits = model(input_tensor)
                probs_tensor = F.softmax(logits, dim=1)
                probs = probs_tensor[0].detach().cpu().numpy()
                argmax_idx = int(np.argmax(probs))

            if args.prediction_rule == "target_threshold":
                pred_idx = apply_threshold_prediction(
                    probs=probs,
                    class_names=class_names,
                    target_class=target_class,
                    threshold=args.positive_threshold,
                )
                pred_rule = f"target_threshold_{target_class}_{args.positive_threshold}"
            else:
                pred_idx = argmax_idx
                pred_rule = "argmax"

            pred_class = class_names[pred_idx]
            pred_conf = float(probs[pred_idx])

            if not should_keep_image(true_class, pred_class, args):
                continue

            target_indices = []
            target_labels = []

            if args.cam_target == "pred":
                target_indices.append(pred_idx)
                target_labels.append(pred_class)
            elif args.cam_target == "true":
                if true_class == "" or true_class not in class_names:
                    print(f"[Warning] {rel_path} 没有有效真实标签，跳过 true CAM。")
                    continue
                true_idx = class_names.index(true_class)
                target_indices.append(true_idx)
                target_labels.append(true_class)
            elif args.cam_target == "class":
                if target_class is None:
                    raise ValueError("cam_target=class 时必须指定 --target_class。")
                target_indices.append(class_names.index(target_class))
                target_labels.append(target_class)
            elif args.cam_target == "both":
                target_indices.append(pred_idx)
                target_labels.append(pred_class)
                if target_class is not None and target_class in class_names:
                    class_idx = class_names.index(target_class)
                    if class_idx != pred_idx:
                        target_indices.append(class_idx)
                        target_labels.append(target_class)

            for cam_idx, cam_label in zip(target_indices, target_labels):
                cam_map, grad_pred_idx, grad_pred_conf, target_score, probs_list = cam_engine(
                    input_tensor=input_tensor,
                    target_index=cam_idx,
                )

                cam_resized = resize_cam_to_image(cam_map, args.img_size)
                heatmap = make_heatmap(cam_resized, colormap_name=args.colormap)
                overlay = overlay_heatmap(rgb_image, heatmap, alpha=args.alpha)

                rel_no_suffix = Path(rel_path).with_suffix("").as_posix()
                output_stem = output_dir / safe_name(true_class if true_class else "unknown") / safe_name(cam_label) / safe_name(rel_no_suffix)

                title = (
                    f"True: {true_class} | Pred: {pred_class} ({pred_conf:.3f}) | "
                    f"CAM target: {cam_label} ({target_score:.3f})"
                )
                paths = save_cam_images(
                    rgb_image=rgb_image,
                    cam_resized=cam_resized,
                    heatmap=heatmap,
                    overlay=overlay,
                    output_stem=output_stem,
                    title=title,
                )

                prob_dict = {
                    f"prob_{class_names[i]}": round(float(probs[i]), 6)
                    for i in range(len(class_names))
                }

                row = {
                    "relative_path": rel_path,
                    "image_name": image_path.name,
                    "true_class": true_class,
                    "pred_rule": pred_rule,
                    "pred_class": pred_class,
                    "pred_confidence": round(float(pred_conf), 6),
                    "argmax_class": class_names[argmax_idx],
                    "argmax_confidence": round(float(probs[argmax_idx]), 6),
                    "cam_target_class": cam_label,
                    "cam_target_score": round(float(target_score), 6),
                    "target_layer": target_layer_name,
                    "panel_path": paths["panel_path"],
                    "overlay_path": paths["overlay_path"],
                    "heatmap_path": paths["heatmap_path"],
                    "original_path": paths["original_path"],
                }
                row.update(prob_dict)
                summary_rows.append(row)

    finally:
        cam_engine.remove_hooks()

    summary_json = output_dir / "cam_summary.json"
    summary_excel = output_dir / "cam_summary.xlsx"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=4)

    df = pd.DataFrame(summary_rows)
    df.to_excel(summary_excel, index=False)

    print("=" * 100)
    print(f"CAM 生成完成，共保存 {len(summary_rows)} 条 CAM 记录。")
    print(f"汇总 JSON : {summary_json}")
    print(f"汇总 Excel: {summary_excel}")
    print(f"输出目录  : {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()


# 运行示例：
#
# 1. 三分类，推荐。每个真实类别取 10 张，同时生成预测类别 CAM 和 IIabc CAM：
# python visualize_ddh_cam.py --num_classes 3 --class_names D_III_IV I IIabc --cam_target both --target_class IIabc --max_per_class 10 --max_images 30
#
# 2. 三分类，只看真实 IIabc 样本，对 IIabc 类生成 CAM：
# python visualize_ddh_cam.py --num_classes 3 --class_names D_III_IV I IIabc --cam_target class --target_class IIabc --only_true_class IIabc --max_images 30
#
# 3. 三分类，结合你前面选出的 IIabc 阈值 0.086 记录预测结果，同时生成 CAM：
# python visualize_ddh_cam.py --num_classes 3 --class_names D_III_IV I IIabc --cam_target both --target_class IIabc --prediction_rule target_threshold --positive_threshold 0.086 --max_per_class 10
#
# 4. 二分类，默认关注 Others：
# python visualize_ddh_cam.py --num_classes 2 --class_names I Others --cam_target both --target_class Others --max_per_class 10
#
# 5. 查看可选 target layer：
# python visualize_ddh_cam.py --num_classes 3 --class_names D_III_IV I IIabc --list_layers
#
# 6. 如果 auto 层报错，可以手动指定某一层，例如：
# python visualize_ddh_cam.py --num_classes 3 --class_names D_III_IV I IIabc --target_layer layers.3.blocks.1.norm2 --cam_target both --target_class IIabc

# python visualize_ddh_cam.py --num_classes 2 --class_names I Others --cam_target pred --target_class Others --prediction_rule target_threshold --positive_threshold 0.188 --only_correct --max_per_class 0 --max_images 0 --output_dir /home/a/vv/DDH/ddh_2cls_cam_correct_threshold



# python visualize_ddh_cam.py --num_classes 3 --class_names D_III_IV I IIabc --cam_target pred --target_class IIabc --prediction_rule target_threshold --positive_threshold 0.086 --only_correct --max_per_class 0 --max_images 0 --output_dir /home/a/vv/DDH/ddh_3cls_cam_correct_IIabc_threshold
