# Standalone Feature Embedding API

该目录可单独复制部署。服务只加载：

- VGG19 前四段笔触特征提取器
- style-token adapter
- 远程骨架提取 API（不可用时可本地 fallback）

不会加载 Stable Diffusion、UNet、VAE、CLIP 或 ControlNet。

## 1. 导出轻量权重

在完整 ControlNet 项目环境中执行：

```bash
python get_stroke_embedding/export_weights.py \
  --ckpt ./training_outputs/train_20260611_021327/checkpoints/last.ckpt \
  --output ./get_stroke_embedding/weights/stroke_embedding.pt
```

权重导出后，可以只复制整个 `get_stroke_embedding` 目录到服务器。

## 2. Docker Compose 构建并启动

```bash
cd get_stroke_embedding
mkdir -p input
CPU_THREADS=4 docker compose up -d --build
docker compose logs -f
```

环境变量：

- `API_PORT=8899`：宿主机端口
- `CPU_THREADS=4`：限制 PyTorch/OpenMP 使用的 CPU 线程数
- `SKELETON_API_URL=...`：骨架服务地址
- `INPUT_DATA_DIR=./input`：需要通过路径读取的图片/字体目录
- `EMBEDDING_API_IMAGE=controlnet-feature-embedding-api:latest`：构建后的镜像名

该镜像基于 `python:3.10-slim`，安装 PyTorch 2.5.1 CPU wheel，完全不依赖 CUDA、显卡或 NVIDIA Container Toolkit。
普通 Python 依赖默认使用清华 PyPI 镜像；PyTorch CPU wheel 使用其官方专用索引。

Dockerfile 会把 `app.py`、特征网络代码和
`weights/stroke_embedding.pt` 一起复制进镜像，因此构建前必须先执行权重导出。

跨服务器调用推荐使用 `image_base64`，避免容器路径差异。

## 3. 接口

```text
GET  /health
POST /embed/skeleton
POST /embed/stroke
POST /similarity/skeleton
POST /similarity/stroke
```

健康检查：

```bash
curl http://127.0.0.1:8899/health
```

本地挂载图片：

```bash
curl -X POST http://127.0.0.1:8899/embed/stroke \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/data/example.png"}'
```

base64 图片：

```bash
IMAGE_BASE64=$(base64 -w 0 ./example.png)
curl -X POST http://127.0.0.1:8899/embed/stroke \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"${IMAGE_BASE64}\"}"
```

TTF 字符（字体需要放入挂载的 `input` 目录）：

```bash
curl -X POST http://127.0.0.1:8899/embed/stroke \
  -H 'Content-Type: application/json' \
  -d '{"ttf_path":"/data/example.ttf","char":"永"}'
```

## 4. 拉伸稳定性测试

比较原图和横向拉伸图的笔触 embedding：

```bash
python test_stroke_stretch_similarity.py ./example.png \
  --api-url http://127.0.0.1:8899 \
  --scale-x 1.5 \
  --scale-y 1.0 \
  --output-dir ./stretch_similarity_result
```

结果目录包含原图、拉伸图、并排对比图和完整 JSON 响应。
