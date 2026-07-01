---
license: apache-2.0
language:
- zh
library_name: diffusers
pipeline_tag: image-to-image
---
<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" width="400"/>
<p>
<p align="center">
          💜 <a href="https://chat.qwen.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit-2511">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf">技术报告</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit-2511/">博客</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit-2511">在线 Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">微信</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp| &nbsp&nbsp <a href="https://github.com/QwenLM/Qwen-Image">Github</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen-Image/edit2511/edit2511big.JPG#center" width="1600"/>
<p>

# 项目简介

`Qwen-Image-Edit-2511` 是 `Qwen-Image-Edit-2509` 的增强版本，重点提升了图像编辑时的一致性表现，并增强了工业设计生成与几何推理能力。

主要改进包括：

- 降低编辑过程中的图像漂移
- 提升人物与角色一致性
- 集成部分社区常用 LoRA 能力
- 增强工业设计场景生成效果
- 增强几何理解与辅助线生成能力

如果你想体验官方在线版本，可以前往 [Qwen Chat](https://chat.qwen.ai/?inputFeature=image_edit) 并选择图像编辑能力。

## 快速开始

### 安装最新版 diffusers

```bash
pip install git+https://github.com/huggingface/diffusers
```

### Python 调用示例

```python
import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
)
print("pipeline loaded")

pipeline.to("cuda")
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

## 能力展示

### 角色一致性增强

`Qwen-Image-Edit-2511` 在人物/角色一致性上有明显提升，可以在保留主体身份特征的前提下完成更具想象力的编辑。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片1.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片2.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片3.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片4.JPG#center)

### 多人一致性增强

相比 `2509` 版本对单主体一致性的提升，`2511` 进一步增强了多人场景的一致性表现，适合将两张人物图融合成自然、统一的群像照片。

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片5.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片6.JPG#center)

### 内置社区 LoRA 效果

社区已经围绕 Qwen-Image 发展出不少高质量 LoRA。`Qwen-Image-Edit-2511` 将部分常见能力整合进基座模型，无需额外加载即可直接使用。

以光照增强为例：

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片7.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片8.JPG#center)

再如视角变化：

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片9.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片10.JPG#center)

### 工业设计场景

项目也针对工程与设计类场景做了增强，例如批量工业产品设计：

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片11.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片12.JPG#center)

以及工业部件材质替换：

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片13.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片14.JPG#center)

### 几何推理增强

模型在几何理解方面也更强，例如可以更稳定地生成辅助构造线、标注线等结果：

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片15.JPG#center)
![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2511/幻灯片16.JPG#center)

## Gradio WebUI

本仓库提供了本地 `Gradio WebUI`，入口文件为 `app.py`，支持：

- 单次图像编辑
- 多图联合编辑
- 批量推理
- 批量结果导出

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动方式

#### 1）单卡启动

适合只有 1 张 GPU，或者只想使用单卡推理：

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
CUDA_VISIBLE_DEVICES=0 python app.py
```

#### 2）双卡启动

当前仓库里的 `manual` 设备映射是按 4 卡写死的，因此 **两卡场景推荐使用 `balanced`**：

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
CUDA_VISIBLE_DEVICES=0,1 DEVICE_MAP=balanced GPU_MEMORY_RESERVE_GB=1 python app.py
```

如果两卡显存比较紧张，可以使用更保守的配置：

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
CUDA_VISIBLE_DEVICES=0,1 DEVICE_MAP=balanced GPU_MEMORY_RESERVE_GB=2 ENABLE_CPU_OFFLOAD=1 ENABLE_ATTENTION_SLICING=1 python app.py
```

#### 3）四卡启动（推荐）

4 卡环境下推荐使用仓库内置的手工分片：

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py
```

#### 4）默认启动

如果你不手动设置环境变量，也可以直接启动：

```bash
cd /mnt/data12/luguiliang/Qwen-Image-Edit-2511
python app.py
```

程序会根据当前可见设备自动选择 CPU / CUDA，并在多卡时根据 `DEVICE_MAP` 配置进行分配。

#### 5）Docker 启动示例

启动容器：

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

进入容器并启动服务：

```bash
docker exec -it QwenImg bash
cd /app
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py
```

启动后，在浏览器访问：

- 本机：`http://127.0.0.1:7860`
- 局域网其他机器：`http://<server-ip>:7860`

### 功能特性

- 支持单张或多张输入图像编辑
- 支持基于 `CSV` / `JSON` 的批量推理
- 批量推理仅展示总进度条，不显示逐文件进度
- 批量推理完成后提供单一 `ZIP` 下载入口
- 单次推理支持自定义输出 `width` / `height`
- 单次推理与远程批量上传文件/结果默认在服务器暂存 7 天
- 在 CUDA 可用时支持自动多卡分配（`device_map="balanced"`）
- 支持 VAE slicing / tiling 与 attention slicing，降低显存压力

## WebUI 使用说明

### 单次推理

在 `单次推理` 页签中：

1. 上传一张或多张输入图片。
2. 在 `Prompt` 中填写编辑指令。
3. 可选填写 `Negative Prompt`。
4. 设置 `Seed`、`推理步数`、`guidance_scale`、`true_cfg_scale` 等参数。
5. 如有需要，可填写 `输出宽度` 和 `输出高度`。
6. 点击 `开始推理`，等待结果生成。

分辨率规则：

- `输出宽度` 和 `输出高度` 都留空：程序会根据输入图像长宽比自动选择分辨率。
- 两个都填写：按指定分辨率输出。
- 只填写一个：请求会被拒绝并提示错误。

说明：

- 更大的分辨率会显著增加显存占用和运行时间。
- 在手工多卡模式下，高分辨率和高步数会进一步拉长总耗时。
- 实际内部尺寸仍可能按模型要求进行对齐。

### 批量推理

在 `批量推理` 页签中：

1. 准备 `CSV` 或 `JSON` 任务文件。
2. 在页面中选择以下任一模式：
   - **服务端本地图片模式**：先把图片目录手动放到项目根目录下的 `batch_inputs/` 中，再上传任务文件。
   - **远程上传模式**：先生成相对图片包根目录的任务文件，再把对应图片目录打成 zip，一起上传任务文件和图片包。
3. 上传任务文件。
4. 如果使用远程上传模式，再上传图片包 `ZIP`。
5. 点击开始批量推理。
6. 等待全部任务执行完成后，下载最终生成的结果 `ZIP`。

说明：

- 批量页只展示一个整体任务状态/总进度区域，不展示逐文件推理进度。
- 批量任务提交后会在后台执行，当前页面可查看总进度，刷新页面后会自动恢复最近一次批量任务状态。
- 批量页完成后只提供一个最终 `ZIP` 下载入口，不再单独展示表格、画廊、CSV、JSON 下载组件。
- 页面会根据所选模式切换输入项：
  - 服务端本地图片模式：只需要上传任务文件
  - 远程上传模式：需要同时上传任务文件和图片包 `ZIP`
- 当前 7 天自动清理规则只覆盖：
  - 单次推理
  - 远程上传批量模式（manifest + zip）
- 服务端本地图片模式当前不纳入 7 天自动清理范围。

当前限制：

- 批量模式当前仍使用默认自动分辨率逻辑，不支持在 manifest 中为每条任务单独指定 `width` / `height`。

## 批量任务生成脚本

仓库包含 `prompt_generation/generate_batch_manifest.py`，可用于扫描图像目录并生成可直接喂给 `批量推理` 页签的 JSON 任务文件。

相关文件：

- `prompt_generation/generate_batch_manifest.py`
- `prompt_generation/prompt_library.yaml`

脚本默认读取 `prompt_generation/prompt_library.yaml` 作为 prompt 库。

### 基础用法

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json --recursive --mode cycle
```

### 当前户外增强场景推荐用法

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

### 使用自定义 YAML Prompt 库

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json --recursive --mode random --library-yaml /path/to/prompt_library.yaml
```

### 服务端本地图片模式推荐生成方式

如果你准备在 WebUI 中使用“服务端本地图片模式”，建议先把图片目录放到项目根目录下的 `batch_inputs/` 中，再生成任务文件：

```bash
python prompt_generation/generate_batch_manifest.py ./batch_inputs/task_001 /path/to/batch_tasks.json \
  --recursive \
  --mode random \
  --image-path-mode project-relative
```

此时任务文件中的 `images` 会写成相对 `batch_inputs/` 的路径，例如 `task_001/a.jpg`。

### 远程上传模式推荐生成方式

如果你准备在 WebUI 中使用“任务文件 + 图片包”的远程上传模式，建议这样生成：

```bash
python prompt_generation/generate_batch_manifest.py /path/to/images /path/to/batch_tasks.json \
  --recursive \
  --mode random \
  --image-path-mode package-relative
```

然后把 `/path/to/images` 这个目录整体打成 zip，并保留内部目录结构。

### `/mnt/data12/data_gen` 示例

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

### Docker 内运行示例

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

### 导出当前目标库为 JSON

```bash
python prompt_generation/generate_batch_manifest.py --export-targets targets.json
```

### 当前目标类别

`prompt_generation/prompt_library.yaml` 当前包含以下类别：

- `宠物与小动物`
- `人物与轻度活动`
- `儿童活动物品`
- `园艺用品与庭院工具`
- `庭院设施与居家杂物`
- `自然杂物与环境干扰`

### 脚本输出说明

脚本输出的 JSON manifest 可直接用于 `批量推理`。每条任务包含以下标准字段：

- `id`
- `prompt`
- `negative_prompt`
- `images`
- `seed`
- `num_inference_steps`
- `guidance_scale`
- `true_cfg_scale`

其中 `images` 是 JSON 数组，`seed` 现在会为每条任务随机生成，而不是递增编号。

### Prompt 生成参数说明

- `prompt_generation/prompt_library.yaml` 控制 prompt 模板、negative prompt、目标库和 `category_ratios`。
- `--library-yaml` 可切换到其他 YAML 库，而无需修改 Python 代码。
- `--image-path-mode project-relative` 会把图片路径写成相对项目根目录 `batch_inputs/` 的路径，适合服务端本地图片模式。
- `--image-path-mode package-relative` 会把图片路径写成相对输入根目录的路径，适合远程上传模式。
- `--local-batch-root` 用于指定 `project-relative` 模式对应的项目内图片根目录，默认是仓库根目录下的 `batch_inputs/`。
- `--image-path-mode absolute` 仅建议用于兼容旧任务文件，不再推荐作为新流程默认方式。
- 当使用 `package-relative` 时，务必从同一个输入根目录打 zip，并保留目录结构。
- `--category-count-min` 和 `--category-count-max` 控制每个目标在候选池中的重复次数。
- `--targets-per-prompt-min` 和 `--targets-per-prompt-max` 控制每条 prompt 混合多少个目标。
- 单条 prompt 现在可以混合多个类别、多个目标。
- `category_ratios` 控制不同类别在全局采样中的倾向。
- `--prompt-seed` 同时控制目标采样、prompt 变体采样和输出 seed 生成；固定该值可复现结果，修改该值会得到新的任务文件。

### 输出目录与 7 天暂存规则

- 单次推理会话目录：`outputs/single/`
- 批量推理会话目录：`outputs/batch/`
- 单次推理现在会为每次请求创建独立会话目录，目录中包含：
  - `inputs/`：上传的原始输入图
  - 生成结果图
  - `metadata.json`
  - 会话状态标记（如 `.completed` / `.failed`）
- 远程上传批量模式会在批量会话目录中保留：
  - `uploads/manifest.*`
  - `uploads/input_package.zip`
  - `input_package/`：解压后的上传图片
  - `batch_results.csv`
  - `batch_results.json`
  - `batch_results.zip`
  - `metadata.json`
  - 会话状态标记
- 单次推理与远程上传批量模式的上传文件、结果文件默认在服务器暂存 7 天，之后自动清理。
- 清理任务会跳过仍处于 `.in_progress` 状态的目录，避免删除正在执行中的任务。
- 服务端本地图片模式当前不纳入 7 天自动清理范围。

## 参数说明

- `Seed`：控制随机性。相同输入 + 相同 seed，通常更容易复现相似结果。
- `推理步数` / `num_inference_steps`：步数越高通常细节和 prompt 服从性越好，但运行时间也会近似线性增加。
- `guidance_scale`：在当前模型路径下实际影响较小，通常建议保留默认值。
- `true_cfg_scale`：控制 prompt 约束强度。值越大通常更贴近 prompt，但也可能增加运行时间，并使结果更激进。
- `Negative Prompt`：可选。很多场景下可以留空或只传空格字符串。

实用建议：

- 快速预览：`num_inference_steps=20~24`，`true_cfg_scale=1.0~2.0`
- 更高质量：`num_inference_steps=32~40`，`true_cfg_scale=3.0~4.0`
- 如果运行太慢，优先降低分辨率和步数。

## 多卡与显存说明

只要 `torch.cuda.is_available()` 为 `True`，程序就会尝试在当前可见 GPU 上运行。

### 可选环境变量

- `CUDA_VISIBLE_DEVICES=0,1,2,3`：指定可见 GPU
- `DEVICE_MAP=manual`：多卡切分策略，当前默认是为 4 卡优化的手工映射
- `GPU_MEMORY_RESERVE_GB=2`：为每张 GPU 预留的显存大小，用于计算 `max_memory`
- `ENABLE_CPU_OFFLOAD=0`：设为 `1` 时，允许在必要时使用 CPU offload
- `CPU_OFFLOAD_MAX_MEMORY=64GiB`：CPU offload 的主机内存预算
- `ENABLE_ATTENTION_SLICING=0`：设为 `1` 时可进一步节省显存，但速度更慢

### 当前 4 卡手工映射

- `text_encoder -> cuda:1`
- `vae / processor / scheduler -> cuda:3`
- `transformer blocks [0:30] -> cuda:0`
- `transformer blocks [30:60] -> cuda:2`

说明：当前环境中的 pipeline loader 在 `from_pretrained()` 时只接受字符串类型的 `device_map`。因此 `DEVICE_MAP=manual` 的实现方式是：先正常加载 pipeline，再在代码里手工把主要模块和 transformer block 分发到固定 GPU。

如果 `DEVICE_MAP` 不是 `manual`，例如 `balanced`，程序会回退到 diffusers 自动放置逻辑。

### 推荐命令

#### 四卡推荐

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=manual GPU_MEMORY_RESERVE_GB=1 python app.py
```

#### 四卡保守模式

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DEVICE_MAP=balanced GPU_MEMORY_RESERVE_GB=2 ENABLE_CPU_OFFLOAD=1 ENABLE_ATTENTION_SLICING=1 python app.py
```

## 批量任务文件格式

批量页支持 `CSV` 和 `JSON` 两种格式。

### CSV 字段

- `id`：可选，任务 ID
- `prompt`：必填，编辑指令
- `negative_prompt`：可选，负向提示词
- `images`：必填，图片路径；多图时使用 `|` 分隔
- `seed`：可选，默认 `0`
- `num_inference_steps`：可选，默认 `40`
- `guidance_scale`：可选，默认 `1.0`
- `true_cfg_scale`：可选，默认 `4.0`

### 路径解析规则

批量页支持两种模式：

- **服务端本地图片模式**：只上传 manifest。
  - 图片必须先放到项目根目录下的 `batch_inputs/` 中。
  - `images` 只能写文件名或相对 `batch_inputs/` 的路径。
  - 不支持绝对路径，也不支持 `..` 越界路径。
- **远程上传模式**：上传 manifest 和 ZIP 图片包。
  - `images` 必须写成相对 ZIP 根目录的路径。
  - 服务端会先把 manifest、ZIP 落到会话目录，再解压 ZIP 并按相对路径解析图片。
  - 不支持绝对路径，也不支持 `..` 越界路径。

说明：

- 最终解析后的图片路径必须位于项目目录控制范围内。
- 两种批量模式都会在后台执行，页面会轮询显示总进度与当前状态；刷新页面后会自动恢复最近一次批量任务状态。
- 只有当批量任务全部完成并打包结束后，前端才会提供最终 `batch_results.zip` 下载入口。
- 若服务重启导致批量任务中断，恢复页面后会显示失败状态，而不是继续显示运行中。
- 批量模式当前不支持为每条任务单独解析 `width` / `height`，仍沿用单次推理中“留空时自动分辨率”的逻辑。

### CSV 示例

```csv
id,prompt,negative_prompt,images,seed,num_inference_steps,guidance_scale,true_cfg_scale
1,A silver robot standing in a flower field., ,examples/input1.png,0,40,1.0,4.0
2,Merge the two people into one travel photo., ,examples/a.png|examples/b.png,42,40,1.0,4.0
```

### JSON 示例

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

## 许可证

Qwen-Image 使用 Apache 2.0 许可证。

## 引用

如果本项目对你有帮助，欢迎引用：

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