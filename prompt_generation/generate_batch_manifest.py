import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPT_GENERATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROMPT_GENERATION_DIR.parent
DEFAULT_LIBRARY_PATH = PROMPT_GENERATION_DIR / "prompt_library.yaml"
DEFAULT_LOCAL_BATCH_ROOT = PROJECT_ROOT / "batch_inputs"


def validate_category_ratios(
    category_ratios: Any,
    target_library: List[Dict[str, Any]],
    yaml_path: Path,
) -> Dict[str, float]:
    if category_ratios is None:
        return {}
    if not isinstance(category_ratios, dict) or not category_ratios:
        raise SystemExit(f"'category_ratios' must be a non-empty mapping in: {yaml_path}")

    known_categories = {str(item["category"]) for item in target_library}
    normalized: Dict[str, float] = {}
    total = 0.0
    for key, value in category_ratios.items():
        if not isinstance(key, str) or not key.strip():
            raise SystemExit(f"Invalid category name in 'category_ratios' of: {yaml_path}")
        if key not in known_categories:
            raise SystemExit(f"Unknown category '{key}' in 'category_ratios' of: {yaml_path}")
        if not isinstance(value, (int, float)):
            raise SystemExit(f"Category ratio for '{key}' must be numeric in: {yaml_path}")
        ratio = float(value)
        if ratio < 0.0 or ratio > 1.0:
            raise SystemExit(f"Category ratio for '{key}' must be between 0 and 1 in: {yaml_path}")
        normalized[key] = ratio
        total += ratio

    if total <= 0.0:
        raise SystemExit(f"Sum of 'category_ratios' must be greater than 0 in: {yaml_path}")
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Sum of 'category_ratios' must be 1.0 in: {yaml_path}")

    return normalized


def load_prompt_library(yaml_path: Path) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, float]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise SystemExit(f"Prompt library YAML must be a mapping: {yaml_path}")

    prompt_template = data.get("prompt_template")
    negative_prompt = data.get("negative_prompt")
    target_library = data.get("target_library")

    if not isinstance(prompt_template, str) or not prompt_template.strip():
        raise SystemExit(f"Missing or invalid 'prompt_template' in: {yaml_path}")
    if not isinstance(negative_prompt, str) or not negative_prompt.strip():
        raise SystemExit(f"Missing or invalid 'negative_prompt' in: {yaml_path}")
    if not isinstance(target_library, list) or not target_library:
        raise SystemExit(f"Missing or invalid 'target_library' in: {yaml_path}")

    required_keys = [
        "category",
        "target_key",
        "target_variants",
        "placement_variants",
        "state_variants",
        "appearance_variants",
    ]
    for index, item in enumerate(target_library, start=1):
        if not isinstance(item, dict):
            raise SystemExit(f"target_library item #{index} must be a mapping in: {yaml_path}")
        for key in required_keys:
            if key not in item:
                raise SystemExit(f"Missing key '{key}' in target_library item #{index} of: {yaml_path}")
            if key.endswith("_variants") and (not isinstance(item[key], list) or not item[key]):
                raise SystemExit(f"'{key}' must be a non-empty list in target_library item #{index} of: {yaml_path}")

    category_ratios = validate_category_ratios(data.get("category_ratios"), target_library, yaml_path)

    return prompt_template, negative_prompt, target_library, category_ratios


def iter_images(folder: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path



def join_descriptions(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}和{parts[1]}"
    return "、".join(parts[:-1]) + f"以及{parts[-1]}"



def build_prompt(targets: List[Dict[str, Any]], rng: random.Random, prompt_template: str) -> str:
    target_descriptions = [rng.choice(target["target_variants"]) for target in targets]
    placement_descriptions = [rng.choice(target["placement_variants"]) for target in targets]
    state_descriptions = [rng.choice(target["state_variants"]) for target in targets]
    appearance_descriptions = [rng.choice(target["appearance_variants"]) for target in targets]

    return prompt_template.format(
        target_description=join_descriptions(target_descriptions),
        placement_description=join_descriptions(placement_descriptions),
        state_description=join_descriptions(state_descriptions),
        appearance_description=join_descriptions(appearance_descriptions),
    )



def expand_targets_by_category_count(
    targets: List[Dict[str, Any]],
    rng: random.Random,
    min_count: int,
    max_count: int,
) -> List[Dict[str, Any]]:
    if min_count < 1 or max_count < 1:
        raise SystemExit("category target count must be >= 1")
    if min_count > max_count:
        raise SystemExit("category target min count cannot be greater than max count")

    expanded_targets: List[Dict[str, Any]] = []
    for target in targets:
        repeat_count = rng.randint(min_count, max_count)
        expanded_targets.extend([target] * repeat_count)
    return expanded_targets



def build_ratio_weighted_targets(
    targets: List[Dict[str, Any]],
    category_ratios: Dict[str, float],
) -> List[Dict[str, Any]]:
    if not category_ratios:
        return list(targets)

    targets_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for target in targets:
        category = str(target["category"])
        targets_by_category.setdefault(category, []).append(target)

    ratio_targets: List[Dict[str, Any]] = []
    scale = 100
    for category, ratio in category_ratios.items():
        category_targets = targets_by_category.get(category, [])
        if not category_targets:
            continue
        repeat_count = max(1, int(round(ratio * scale)))
        for target in category_targets:
            ratio_targets.extend([target] * repeat_count)

    return ratio_targets or list(targets)



def select_targets(
    index: int,
    targets: List[Dict[str, Any]],
    mode: str,
    rng: random.Random,
    min_targets_per_prompt: int,
    max_targets_per_prompt: int,
) -> List[Dict[str, Any]]:
    if min_targets_per_prompt < 1 or max_targets_per_prompt < 1:
        raise SystemExit("targets per prompt must be >= 1")
    if min_targets_per_prompt > max_targets_per_prompt:
        raise SystemExit("targets per prompt min cannot be greater than max")

    desired_count = rng.randint(min_targets_per_prompt, max_targets_per_prompt)
    desired_count = min(desired_count, len(targets))
    if desired_count < 1:
        raise SystemExit("No targets available for prompt composition")

    selected: List[Dict[str, Any]] = []
    used_keys = set()
    cursor = index
    attempts = 0
    max_attempts = max(len(targets) * 4, desired_count * 4)

    while len(selected) < desired_count and attempts < max_attempts:
        candidate = rng.choice(targets) if mode == "random" else targets[cursor % len(targets)]
        cursor += 1
        attempts += 1
        target_key = str(candidate["target_key"])
        if target_key in used_keys:
            continue
        used_keys.add(target_key)
        selected.append(candidate)

    if len(selected) < desired_count:
        for candidate in targets:
            target_key = str(candidate["target_key"])
            if target_key in used_keys:
                continue
            used_keys.add(target_key)
            selected.append(candidate)
            if len(selected) >= desired_count:
                break

    return selected



def serialize_image_path(image_path: Path, input_root: Path, path_mode: str, local_batch_root: Path) -> str:
    if path_mode == "absolute":
        return str(image_path)
    if path_mode == "package-relative":
        return image_path.relative_to(input_root).as_posix()
    if path_mode == "project-relative":
        try:
            return image_path.relative_to(local_batch_root).as_posix()
        except ValueError as exc:
            raise SystemExit(
                "project-relative 模式要求输入图片目录位于本地批处理根目录下: "
                f"{local_batch_root}"
            ) from exc
    raise SystemExit(f"Unsupported image path mode: {path_mode}")



def to_output_rows(
    image_paths: List[Path],
    targets: List[Dict[str, Any]],
    prompt_template: str,
    negative_prompt: str,
    mode: str,
    prompt_seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
    category_count_min: int,
    category_count_max: int,
    category_ratios: Dict[str, float],
    min_targets_per_prompt: int,
    max_targets_per_prompt: int,
    input_root: Path,
    image_path_mode: str,
    local_batch_root: Path,
) -> List[dict]:
    rng = random.Random(prompt_seed)
    ratio_weighted_targets = build_ratio_weighted_targets(targets, category_ratios)
    expanded_targets = expand_targets_by_category_count(
        targets=ratio_weighted_targets,
        rng=rng,
        min_count=category_count_min,
        max_count=category_count_max,
    )
    if not expanded_targets:
        raise SystemExit("No targets available after category expansion")

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        selected_targets = select_targets(
            index=index - 1,
            targets=expanded_targets,
            mode=mode,
            rng=rng,
            min_targets_per_prompt=min_targets_per_prompt,
            max_targets_per_prompt=max_targets_per_prompt,
        )
        prompt = build_prompt(selected_targets, rng, prompt_template)
        rows.append(
            {
                "id": str(index),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "images": [
                    serialize_image_path(
                        image_path,
                        input_root=input_root,
                        path_mode=image_path_mode,
                        local_batch_root=local_batch_root,
                    )
                ],
                "seed": rng.randint(0, 2**31 - 1),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "true_cfg_scale": true_cfg_scale,
            }
        )
    return rows



def write_manifest_json(rows: List[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)



def write_targets_json(output_path: Path, targets: List[Dict[str, Any]]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate batch manifest for outdoor scene augmentation.")
    parser.add_argument("input_dir", type=Path, nargs="?", help="Folder containing source images.")
    parser.add_argument("output_json", type=Path, nargs="?", help="Path to output JSON manifest.")
    parser.add_argument("--recursive", action="store_true", help="Scan images recursively.")
    parser.add_argument("--mode", choices=["cycle", "random"], default="cycle", help="How to assign targets to images.")
    parser.add_argument("--prompt-seed", type=int, default=1234, help="Random seed used for target, prompt variation, and output seed sampling.")
    parser.add_argument("--num-inference-steps", type=int, default=40, help="Value written to manifest.")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="Value written to manifest.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="Value written to manifest.")
    parser.add_argument("--category-count-min", type=int, default=1, help="Minimum repeat count for each target/category when building the target pool.")
    parser.add_argument("--category-count-max", type=int, default=1, help="Maximum repeat count for each target/category when building the target pool.")
    parser.add_argument("--targets-per-prompt-min", type=int, default=2, help="Minimum number of targets combined into one prompt.")
    parser.add_argument("--targets-per-prompt-max", type=int, default=4, help="Maximum number of targets combined into one prompt.")
    parser.add_argument(
        "--image-path-mode",
        choices=["project-relative", "package-relative", "absolute"],
        default="project-relative",
        help="How to write image paths into the manifest.",
    )
    parser.add_argument(
        "--local-batch-root",
        type=Path,
        default=DEFAULT_LOCAL_BATCH_ROOT,
        help="Project-local batch image root used by project-relative mode.",
    )
    parser.add_argument("--library-yaml", type=Path, default=DEFAULT_LIBRARY_PATH, help="Path to prompt library YAML file.")
    parser.add_argument("--export-targets", type=Path, help="Export the loaded target library as JSON and exit.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    library_yaml = args.library_yaml.resolve()
    if not library_yaml.exists() or not library_yaml.is_file():
        raise SystemExit(f"Prompt library YAML does not exist: {library_yaml}")

    prompt_template, negative_prompt, target_library, category_ratios = load_prompt_library(library_yaml)

    if args.export_targets:
        output_path = args.export_targets.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_targets_json(output_path, target_library)
        print(f"Exported target library -> {output_path}")
        return

    if args.input_dir is None or args.output_json is None:
        raise SystemExit("input_dir and output_json are required unless --export-targets is used")

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a folder: {input_dir}")

    local_batch_root = args.local_batch_root.resolve()

    image_paths = list(iter_images(input_dir, recursive=args.recursive))
    if not image_paths:
        raise SystemExit(f"No images found in: {input_dir}")

    rows = to_output_rows(
        image_paths=image_paths,
        targets=target_library,
        prompt_template=prompt_template,
        negative_prompt=negative_prompt,
        mode=args.mode,
        prompt_seed=args.prompt_seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        category_count_min=args.category_count_min,
        category_count_max=args.category_count_max,
        category_ratios=category_ratios,
        min_targets_per_prompt=args.targets_per_prompt_min,
        max_targets_per_prompt=args.targets_per_prompt_max,
        input_root=input_dir,
        image_path_mode=args.image_path_mode,
        local_batch_root=local_batch_root,
    )
    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_json(rows, output_path)

    print(f"Generated {len(rows)} tasks -> {output_path}")
    print(f"Target mode: {args.mode}")
    print(f"Image path mode: {args.image_path_mode}")
    if args.image_path_mode == "project-relative":
        print(f"Local batch root: {local_batch_root}")
    print(f"Prompt library: {library_yaml}")
    print(f"Target library size: {len(target_library)}")
    if category_ratios:
        print(f"Category ratios: {category_ratios}")
    print(f"Per-target repeat count: {args.category_count_min}-{args.category_count_max}")
    print(f"Targets per prompt: {args.targets_per_prompt_min}-{args.targets_per_prompt_max}")


if __name__ == "__main__":
    main()
