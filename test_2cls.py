import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import timm
from torchvision import transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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

    image_paths = sorted(image_paths)
    return image_paths


def clean_state_dict(state_dict):
    """
    兼容 DataParallel / DDP / torch.compile 保存的权重前缀。
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        new_state_dict[k] = v
    return new_state_dict


def load_checkpoint(model_path):
    ckpt = torch.load(model_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)
    return state_dict, ckpt


def infer_num_classes(state_dict):
    """
    从分类头权重中自动推断类别数。
    常见分类头名称包括 head.fc.weight, head.weight, classifier.weight 等。
    """
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

    raise ValueError("无法从 pth 中自动推断 num_classes，请手动指定 --num_classes")


def build_transform(img_size=384):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def predict_one_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence, probs[0].cpu().tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_root",
        type=str,
        default=r"DDH_grouped/DDH_2cls/test",
        help="DDH 二分类测试集路径，例如 DDH_grouped/DDH_2cls/test"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=r"runs_transformer_DDH_2class/swinv2_large/best.pth",
        help="保存的 best.pth 路径"
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=r"runs_transformer_DDH_2class/swinv2_large/ddh_2cls_test_predictions.json",
        help="预测结果 json 保存路径"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        help="timm 模型名称，需要和训练时保持一致"
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="输入图像尺寸，需要和训练时保持一致"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="类别数。默认从 pth 自动推断"
    )

    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="类别名称，例如 --class_names I Others。顺序必须和训练时 class_to_idx 一致"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_root = Path(args.test_root)
    model_path = Path(args.model_path)
    output_json = Path(args.output_json)

    if not test_root.exists():
        raise FileNotFoundError(f"测试集路径不存在: {test_root}")

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    state_dict, ckpt = load_checkpoint(model_path)

    if args.num_classes is None:
        num_classes = infer_num_classes(state_dict)
    else:
        num_classes = args.num_classes

    if args.class_names is not None:
        class_names = args.class_names
    else:
        # 优先尝试从 checkpoint 中读取类别映射
        if isinstance(ckpt, dict) and "class_to_idx" in ckpt:
            class_to_idx = ckpt["class_to_idx"]
            class_names = [None] * len(class_to_idx)
            for name, idx in class_to_idx.items():
                class_names[idx] = name
        else:
            # 如果 pth 里没有 class_to_idx，就根据 test 文件夹子目录推断
            # ImageFolder 默认是按字母顺序排序，因此二分类一般是 ['I', 'Others']
            class_names = sorted([
                p.name for p in test_root.iterdir()
                if p.is_dir()
            ])

    if len(class_names) != num_classes:
        raise ValueError(
            f"类别名称数量和模型类别数不一致: "
            f"len(class_names)={len(class_names)}, num_classes={num_classes}。"
            f"请手动指定，例如 --class_names I Others"
        )

    print("=" * 80)
    print(f"Device       : {device}")
    print(f"Model name   : {args.model_name}")
    print(f"Model path   : {model_path}")
    print(f"Test root    : {test_root}")
    print(f"Num classes  : {num_classes}")
    print(f"Class names  : {class_names}")
    print(f"Output json  : {output_json}")
    print("=" * 80)

    model = timm.create_model(
        args.model_name,
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

    transform = build_transform(args.img_size)

    image_paths = collect_images(test_root)

    if len(image_paths) == 0:
        raise RuntimeError(f"没有在测试集目录中找到图片: {test_root}")

    results = {}

    for image_path in tqdm(image_paths, desc="Predicting"):
        rel_path = image_path.relative_to(test_root).as_posix()

        pred_idx, confidence, probs = predict_one_image(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device
        )

        pred_class = class_names[pred_idx]

        results[rel_path] = {
            "image_name": image_path.name,
            "pred_index": pred_idx,
            "pred_class": pred_class,
            "confidence": round(float(confidence), 6),
            "probabilities": {
                class_names[i]: round(float(probs[i]), 6)
                for i in range(num_classes)
            }
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("=" * 80)
    print(f"预测完成，共预测 {len(results)} 张图片")
    print(f"结果已保存到: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# python test_2cls.py --test_root /home/a/vv/DDH/DDH_grouped/DDH_2cls/test --model_path /home/a/vv/DDH/runs_transformer_DDH_2class/swinv2_large/best.pth --output_json /home/a/vv/DDH/ddh_2cls_test_predictions.json --class_names I Others
