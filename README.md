---
license: apache-2.0
language:
- en
- zh
library_name: diffusers
pipeline_tag: image-to-image
---
<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" width="400"/>
<p>
<p align="center">
          💜 <a href="https://chat.qwen.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2511">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit-2511/">Blog</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp| &nbsp&nbsp <a href="https://github.com/QwenLM/Qwen-Image">Github</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen-Image/edit2511/edit2511big.JPG#center" width="1600"/>
<p>


# Introduction

We are excited to introduce Qwen-Image-Edit-2511, an enhanced version over Qwen-Image-Edit-2509, featuring multiple improvements—including notably better consistency. To try out the latest model, please visit [Qwen Chat](https://chat.qwen.ai/?inputFeature=image_edit) and select the Image Editing feature.

Key enhancements in Qwen-Image-Edit-2511 include: mitigate image drift, improved character consistency，integrated LoRA capabilities， enhanced industrial design generation, and strengthened geometric reasoning ability.


## Quick Start

Install the latest version of diffusers
```
pip install git+https://github.com/huggingface/diffusers
```

The following contains a code snippet illustrating how to use `Qwen-Image-Edit-2511`:

```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("input1.png")
image2 = Image.open("input2.png")
prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit_2511.png")
    print("image saved at", os.path.abspath("output_image_edit_2511.png"))

```

## Showcase

**Qwen-Image-Edit-2511 Enhances Character Consistency**
In Qwen-Image-Edit-2511, character consistency has been significantly improved. The model can perform imaginative edits based on an input portrait while preserving the identity and visual characteristics of the subject.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片1.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片2.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片3.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片4.JPG#center)

**Improved Multi-Person Consistency**
While Qwen-Image-Edit-2509 already improved consistency for single-subject editing, Qwen-Image-Edit-2511 further enhances consistency in multi-person group photos—enabling high-fidelity fusion of two separate person images into a coherent group shot:
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片5.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片6.JPG#center)

**Built-in Support for Community-Created LoRAs**
Since Qwen-Image-Edit’s release, the community has developed many creative and high-quality LoRAs—greatly expanding its expressive potential. Qwen-Image-Edit-2511 integrates selected popular LoRAs directly into the base model, unlocking their effects without extra tuning.

For example, Lighting Enhancement LoRA
Realistic lighting control is now achievable out-of-the-box:
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片7.JPG#center)

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片8.JPG#center)

Another example, generating new viewpoints can now be done directly with the base model:

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片9.JPG#center)

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片10.JPG#center)

**Industrial Design Applications**

We’ve paid special attention to practical engineering scenarios—for instance, batch industrial product design:


![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片11.JPG#center)

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片12.JPG#center)

…and material replacement for industrial components:
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片13.JPG#center)

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片14.JPG#center)

**Enhanced Geometric Reasoning**
Qwen-Image-Edit-2511 introduces stronger geometric reasoning capability—e.g., directly generating auxiliary construction lines for design or annotation purposes:


![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片15.JPG#center)

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片16.JPG#center)

That wraps up the major updates in Qwen-Image-Edit-2511.
Enjoy exploring the new capabilities! 🎉

## Gradio WebUI

This repository now includes a local Gradio WebUI in `app.py` for both single-image editing and batch inference.

### Install

```bash
pip install -r requirements.txt
```

### Launch

Local launch:

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
python app.py
```

Recommended multi-GPU launch:

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py
```

Docker example:

```bash
docker run -d --name QwenImg \
  --gpus all \
  --shm-size=15g \
  --network host \
  -v /mnt/data12/luguiliang/Qwen-Image-Edit-2511:/app \
  -v /mnt:/mnt \
  -w /app \
  --entrypoint tail \
  torch2.9.1_cu130_py312_ubuntu22.04:latest -f /dev/null
```

Start the app inside the container:

```bash
docker exec -it QwenImg bash
cd /app
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py
```

Then open `http://127.0.0.1:7860` in your browser, or use `http://<server-ip>:7860` from another machine on the same network.

### Features

- Single inference with one or multiple input images
- Batch inference from a CSV or JSON manifest
- Result preview in the browser
- Export batch outputs as `CSV`, `JSON`, or `ZIP`
- Single inference supports custom output `width` and `height`
- Automatic multi-GPU sharding with `device_map="balanced"` when CUDA is available
- VAE slicing/tiling and attention slicing to reduce GPU memory pressure

### WebUI Usage

#### Single Inference

On the `单次推理` tab:

1. Upload one or multiple input images.
2. Enter the edit prompt in `Prompt`.
3. Optionally fill `Negative Prompt`.
4. Set inference parameters such as `Seed`, `推理步数`, `guidance_scale`, and `true_cfg_scale`.
5. Optionally set `输出宽度` and `输出高度`.
6. Click `开始生成` and wait for the result preview.

Resolution rules:
- Leave both `输出宽度` and `输出高度` empty: the app automatically chooses a resolution based on the input image aspect ratio.
- Fill both `输出宽度` and `输出高度`: the app uses the specified output resolution.
- Fill only one of them: the request is rejected and an error is shown.

Notes:
- Larger resolutions require more GPU memory and are noticeably slower.
- In manual multi-GPU mode, high resolution plus high step count can significantly increase total runtime.
- The final internal size may still be aligned to values required by the model implementation.

#### Batch Inference

On the `批量推理` tab:

1. Prepare a `CSV` or `JSON` manifest.
2. Choose one of the following modes:
   - Local mode: make sure every row/item contains image path(s) readable by the machine running `app.py`, then upload only the manifest.
   - Remote upload mode: generate a manifest with package-relative image paths, zip the corresponding image folder while preserving its internal structure, then upload both the manifest and the image package.
3. Upload the manifest file.
4. Optionally upload the image package `ZIP` if using remote upload mode.
5. Start batch inference and wait for all tasks to finish.
6. Review the generated table/gallery and download `CSV`, `JSON`, or `ZIP` outputs if needed.

Current limitation:
- Batch inference currently uses the default automatic output resolution logic and does not yet expose per-item `width` / `height` fields in the manifest.

#### Batch prompt generation helper

This repository also includes `prompt_generation/generate_batch_manifest.py` for outdoor scene data augmentation. It scans an image folder and generates a JSON manifest directly compatible with the `批量推理` tab in `app.py`.

Prompt-generation assets are now grouped in `prompt_generation/`:
- `prompt_generation/generate_batch_manifest.py`
- `prompt_generation/prompt_library.yaml`

The script now uses the YAML prompt library in `prompt_generation/prompt_library.yaml` by default.

Typical usage:

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json --recursive --mode cycle
```

Recommended command for the current outdoor augmentation setup:

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json \
  --recursive \
  --mode random \
  --prompt-seed 1234 \
  --category-count-min 1 \
  --category-count-max 5 \
  --targets-per-prompt-min 2 \
  --targets-per-prompt-max 4 \
  --num-inference-steps 40 \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0
```

Use a custom YAML prompt library:

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json --recursive --mode random --library-yaml /path/to/prompt_library.yaml
```

Generate a manifest for remote upload mode (`任务文件 + 图片包`) with package-relative image paths:

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json \
  --recursive \
  --mode random \
  --image-path-mode package-relative
```

Example for images under `/mnt/data12/data_gen`:

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
python prompt_generation/generate_batch_manifest.py /mnt/data12/data_gen /mnt/data12/data_gen/batch_tasks.json \
  --recursive \
  --mode random \
  --prompt-seed 1234 \
  --category-count-min 1 \
  --category-count-max 5 \
  --targets-per-prompt-min 2 \
  --targets-per-prompt-max 4 \
  --num-inference-steps 40 \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0
```

Run the same command inside Docker:

```bash
docker exec -it QwenImg bash
cd /app
python prompt_generation/generate_batch_manifest.py /mnt/data12/data_gen /mnt/data12/data_gen/batch_tasks.json \
  --recursive \
  --mode random \
  --prompt-seed 1234 \
  --category-count-min 1 \
  --category-count-max 5 \
  --targets-per-prompt-min 2 \
  --targets-per-prompt-max 4 \
  --num-inference-steps 40 \
  --guidance-scale 1.0 \
  --true-cfg-scale 4.0
```

Export the current YAML target library as JSON:

```bash
python prompt_generation/generate_batch_manifest.py --export-targets targets.json
```

Current target groups in `prompt_generation/prompt_library.yaml` include:
- `宠物与小动物`
- `人物与轻度活动`
- `儿童活动物品`
- `园艺用品与庭院工具`
- `庭院设施与居家杂物`
- `自然杂物与环境干扰`

The script writes a JSON manifest directly compatible with the `批量推理` tab. Each task uses the standard fields `id`, `prompt`, `negative_prompt`, `images`, `seed`, `num_inference_steps`, `guidance_scale`, and `true_cfg_scale`, where `images` is a JSON array. The generated `seed` is now random per task instead of incremental.

Notes for prompt generation:
- `prompt_generation/prompt_library.yaml` controls the prompt template, negative prompt, target library, and `category_ratios`.
- `--library-yaml` lets you switch to another YAML file without editing Python code.
- `--image-path-mode absolute` keeps the current local-path behavior.
- `--image-path-mode package-relative` writes image paths relative to the input image root, which is the recommended mode for remote upload (`任务文件 + 图片包`).
- When using `package-relative`, create the ZIP from the same input root used to generate the manifest and preserve the internal folder structure.
- `--category-count-min` and `--category-count-max` control how many times each target entry is randomly repeated in the candidate pool before assignment.
- `--targets-per-prompt-min` and `--targets-per-prompt-max` control how many targets are combined into one prompt.
- A single prompt can now mix multiple targets and multiple categories.
- `category_ratios` controls the global tendency of category sampling across generated tasks.
- `--prompt-seed` controls target sampling, prompt variation sampling, and random manifest seed generation. Keeping the same value makes the output reproducible; changing it gives a different manifest.

Batch output locations:
- Single inference images: `outputs/single/`
- Batch inference session folders: `outputs/batch/`
- Each batch run creates a new folder such as `outputs/batch/qwen_image_edit_batch_xxxxxxxx/`
- Generated images, `batch_results.csv`, `batch_results.json`, and `batch_results.zip` are saved in that session folder

### Parameter Notes

- `Seed`: controls randomness. Using the same seed and the same inputs usually makes results more reproducible.
- `推理步数` / `num_inference_steps`: more steps usually improve detail and prompt adherence, but runtime grows almost linearly.
- `guidance_scale`: currently has little practical effect for this app and model path; keeping the default value is recommended.
- `true_cfg_scale`: stronger prompt conditioning. Larger values may improve prompt adherence, but also increase runtime and sometimes make outputs more aggressive.
- `Negative Prompt`: optional. In many cases it can be left empty or as a blank string.

Practical recommendations:
- Faster preview: `num_inference_steps=20~24`, `true_cfg_scale=1.0~2.0`
- Higher quality: `num_inference_steps=32~40`, `true_cfg_scale=3.0~4.0`
- If runtime is too slow, first reduce resolution and step count.

### Multi-GPU Notes

The app automatically spreads the pipeline across all visible GPUs when `torch.cuda.is_available()` is `True`.

Optional environment variables:
- `CUDA_VISIBLE_DEVICES=0,1,2,3`: choose which GPUs are visible
- `DEVICE_MAP=manual`: multi-GPU sharding strategy, default is the manual map tuned for 4 GPUs
- `GPU_MEMORY_RESERVE_GB=2`: reserve memory on each GPU before computing `max_memory`
- `ENABLE_CPU_OFFLOAD=0`: when set to `1`, allow CPU offload as a last resort
- `CPU_OFFLOAD_MAX_MEMORY=64GiB`: host RAM budget for CPU offload
- `ENABLE_ATTENTION_SLICING=0`: when set to `1`, reduce memory at the cost of slower inference

Manual device map for 4 GPUs:
- `text_encoder -> cuda:1`
- `vae / processor / scheduler -> cuda:3`
- `transformer blocks [0:30] -> cuda:0`
- `transformer blocks [30:60] -> cuda:2`

Note: the pipeline loader in this environment only accepts `device_map` as a string during `from_pretrained()`. Therefore `DEVICE_MAP=manual` is implemented by loading the pipeline normally first, then dispatching major modules and transformer blocks to fixed GPUs in code.

If `DEVICE_MAP` is set to something else such as `balanced`, the app falls back to diffusers automatic placement.

Recommended:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py
```

Fallback:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=balanced GPU_MEMORY_RESERVE_GB=2 ENABLE_CPU_OFFLOAD=1 ENABLE_ATTENTION_SLICING=1 python app.py
```

### Batch Manifest Format

The batch tab accepts either `CSV` or `JSON`.

CSV fields:
- `id`: optional task id
- `prompt`: required prompt text
- `negative_prompt`: optional negative prompt
- `images`: required image path(s); use `|` to separate multiple image paths
- `seed`: optional, default `0`
- `num_inference_steps`: optional, default `40`
- `guidance_scale`: optional, default `1.0`
- `true_cfg_scale`: optional, default `4.0`

Notes:
- The batch tab supports two modes:
  - Local mode: upload only the manifest. Absolute paths are used directly. Relative paths are resolved against the manifest directory first, then fall back to the project directory for backward compatibility.
  - Remote upload mode: upload the manifest together with a ZIP image package. In this mode, `images` must use paths relative to the ZIP root.
- Image paths in the manifest must be readable by the machine running `app.py` after resolution.
- Batch mode currently does not parse per-item `width` and `height`; it uses the same automatic resolution behavior as single inference with empty width/height.

Example CSV:

```csv
id,prompt,negative_prompt,images,seed,num_inference_steps,guidance_scale,true_cfg_scale
1,A silver robot standing in a flower field., ,examples/input1.png,0,40,1.0,4.0
2,Merge the two people into one travel photo., ,examples/a.png|examples/b.png,42,40,1.0,4.0
```

Example JSON:

```json
[
  {
    "id": "1",
    "prompt": "请基于原图进行真实照片级局部编辑，在带有草地或路面的真实户外场景中自然添加一只小型宠物狗和一个足球以及一段细长树枝，不要改变其他内容。",
    "negative_prompt": "不要重绘整张图，不要改变原图景深，不要改变草地、路面、建筑、天空、背景、车辆和人物，不要新增未指定物体，不要悬浮，不要贴纸感，不要卡通感，不要塑料感，不要棚拍光感，不要过度锐化，不要异常高清边缘，不要错误透视，不要错误比例。",
    "images": ["examples/input1.png"],
    "seed": 148392017,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "true_cfg_scale": 4.0
  }
]
```

## License Agreement

Qwen-Image is licensed under Apache 2.0. 

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```


CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py