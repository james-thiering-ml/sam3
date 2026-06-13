# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install

```bash
pip install -e ".[dev,train]"
```

### Lint and Format

```bash
ufmt format sam3 scripts   # auto-format
ufmt check .               # check without modifying (what CI runs)
```

### Tests

```bash
pytest                                                        # all tests
pytest test/test_io_utils.py                                  # specific file
pytest test/test_io_utils.py::TestLoadVideoFramesRouting      # specific class
pytest test/test_io_utils.py::TestLoadVideoFramesRouting::test_mp4_extension_routes_to_video_loader  # specific test
pytest sam3/perflib/tests/tests.py                            # perflib tests
```

### Type Checking

```bash
mypy sam3
```

## Architecture

SAM 3 is a foundation model for open-vocabulary promptable segmentation in images and videos. It has two main components that share a vision backbone:

- **Detector** (DETR-based): detects and segments object instances given text, bounding box, point, or visual-exemplar prompts
- **Tracker** (SAM 2-inspired): propagates and tracks objects across video frames

**Backbone**: ViT-H (1024 hidden dims, 32 layers, 16 heads, 1008×1008 input, 14px patches, RoPE positional embeddings)

### Public API surface

**Image inference** (`sam3/model/sam3_image_processor.py`):
`Sam3Processor` — `set_image()` → `set_text_prompt()` / `add_geometric_prompt()` → masks, boxes, scores

**Video inference** (`sam3/model/sam3_tracking_predictor.py`):
Session-based API; frame-by-frame propagation with interactive refinement.

**Model factory** (`sam3/model_builder.py`):
`build_sam3_image_model()`, `build_sam3_predictor()` — exported from `sam3/__init__.py`

### Module map

| Path | Purpose |
|------|---------|
| `sam3/model/sam3_image.py` | Core `Sam3Image` nn.Module (backbone + heads) |
| `sam3/model/sam3_image_processor.py` | Public image inference API (`Sam3Processor`) |
| `sam3/model/sam3_video_predictor.py` | Public video entry point |
| `sam3/model/sam3_tracking_predictor.py` | Tracker implementation |
| `sam3/model/vitdet.py` | ViT backbone |
| `sam3/model/encoder.py` | Vision-language encoder with fusion |
| `sam3/model/maskformer_segmentation.py` | Pixel decoder + segmentation head |
| `sam3/model/geometry_encoders.py` | Prompt encoders (boxes, points, masks) |
| `sam3/model/io_utils.py` | Video/image frame loading |
| `sam3/perflib/` | Performance-critical ops (masks↔boxes, NMS, Triton kernels) |
| `sam3/agent/` | LLM-augmented agent for complex reasoning over detections |
| `sam3/train/` | Hydra-based training framework |
| `sam3/eval/` | Evaluation utilities (cgF1, COCO, HOTA) |

### Training

Entry point: `sam3/train/train.py` (Hydra config-driven).

```bash
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4
```

Configs live in `sam3/train/configs/` and cover: Roboflow 100-VL fine-tuning, ODinW13 few-shot, SA-Co/Gold/Silver/VEval evaluation sweeps. The trainer supports submitit for cluster submission (`--use-cluster 1`).

### Primary evaluation metrics

- **cgF1**: concept-guided F1 (main metric for concept segmentation benchmarks)
- **pHOTA**: pose-aware tracking metric (video)
- **AP**: standard COCO detection metric

### Model checkpoints

Checkpoints are gated on Hugging Face (`facebook/sam3.1`). Run `hf auth login` before first use.
