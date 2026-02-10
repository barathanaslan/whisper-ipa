# Mac Studio M3 Ultra - Deep Learning Reference Guide

**System:** Mac Studio M3 Ultra | 96GB Unified Memory | MPS Backend  
**Purpose:** Project-agnostic reference for all deep learning projects  
**Maintenance:** Update with findings from each project

---

## Part 1: Copy This to Every New Project

```python
"""
==============================================================================
MAC STUDIO M3 ULTRA - PROJECT CONFIGURATION
==============================================================================
System: Mac Studio M3 Ultra, 96GB Unified Memory
Backend: MPS (Metal Performance Shaders) via PyTorch
Goal: Always GPU acceleration, never CPU fallback

CONSTRAINTS:
- NO ROI operations (torchvision.ops.roi_align, roi_pool)
- NO distributed training
- NO float64/double precision (use float32 only)
- NO CUDA code (Mac doesn't support CUDA)
- DataLoader num_workers=0 (Mac requirement)

For architecture compatibility, see MAC_STUDIO_ML_REFERENCE.md
==============================================================================
"""
import os
import torch

# Force MPS, disable CPU fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# Verify MPS
assert torch.backends.mps.is_available(), "MPS not available"
assert torch.backends.mps.is_built(), "PyTorch not built with MPS"

device = torch.device("mps")

print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
```

**Copy the above code block to the top of every notebook/script.**

---

## Part 2: PyTorch Installation

```bash
# Recommended: Latest stable
conda install pytorch torchvision torchaudio -c pytorch

# Alternative: Nightly (for latest MPS fixes)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Verify installation
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

---

## Part 3: Architecture Compatibility

### ✅ Works on MPS

#### Image Classification (All Fully Supported)

```python
import torchvision.models as models

# Examples - all work on MPS:
model = models.resnet50(weights="DEFAULT").to(device)
model = models.efficientnet_b0(weights="DEFAULT").to(device)
model = models.vit_b_16(weights="DEFAULT").to(device)
model = models.convnext_tiny(weights="DEFAULT").to(device)
```

**Supported architectures:**

- ResNet, EfficientNet, Vision Transformer (ViT)
- ConvNeXt, DenseNet, MobileNet, RegNet

#### Semantic Segmentation (All Fully Supported)

```python
import torchvision.models.segmentation as seg

# Examples - all work on MPS:
model = seg.deeplabv3_resnet50(weights="DEFAULT").to(device)
model = seg.fcn_resnet50(weights="DEFAULT").to(device)
```

**Supported architectures:**

- DeepLabV3, DeepLabV3+, U-Net, FCN, PSPNet, SegFormer

#### Object Detection (YOLO Works, RCNN Does Not)

```python
# ✅ WORKS: YOLO
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=100, device='mps')

# ❌ DOES NOT WORK: Faster RCNN (uses ROI operations)
# import torchvision.models.detection as det
# model = det.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # Will fail on MPS
```

**What works:**

- ✅ YOLO (v5, v7, v8, v11 via Ultralytics)
- ✅ EfficientDet, DETR (transformer-based)

**What doesn't work:**

- ❌ Faster RCNN, Mask RCNN (ROI operations not supported)

#### Instance Segmentation

- ✅ YOLACT, SOLOv2, CondInst
- ❌ Mask RCNN (ROI operations)

#### Transformers / NLP

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased").to(device)
```

**Supported:**

- BERT, RoBERTa, DistilBERT, GPT-2, T5, BART
- Vision Transformers (ViT, DeiT, Swin)
- ⚠️ Large models (>7B params) may be slow

#### Common Operations (All Supported)

- Conv2D, Conv3D, Linear layers
- BatchNorm, LayerNorm, GroupNorm
- ReLU, GELU, SiLU, Sigmoid, Tanh
- MaxPool, AvgPool, AdaptiveAvgPool
- Dropout, CrossEntropyLoss, BCELoss, MSELoss
- Adam, AdamW, SGD optimizers
- Attention mechanisms, Einsum

### ❌ Does Not Work on MPS

#### Critical Limitation: ROI Operations

```python
# These operations will fail:
import torchvision.ops as ops
# ops.roi_align()  # NotImplementedError
# ops.roi_pool()   # NotImplementedError
```

**Impact:** Cannot use Faster RCNN, Mask RCNN, or any RCNN-based architecture

**Solution:** Use YOLO, EfficientDet, or DETR for object detection instead

#### Other Limitations

- ❌ Float64/double precision (only float32 supported)
- ❌ Some advanced indexing operations
- ❌ Distributed training (single GPU only)
- ⚠️ Very large batch sizes may cause memory errors

---

## Part 4: Testing New Architectures

### Quick Compatibility Test Function

```python
def test_mps_compatibility(model_fn, input_shape=(1, 3, 224, 224)):
    """
    Test if a model architecture works on MPS.

    Args:
        model_fn: Function that returns the model
        input_shape: Input tensor shape

    Returns:
        (success: bool, message: str)
    """
    device = torch.device("mps")

    try:
        # Create model and move to MPS
        model = model_fn()
        model.to(device)
        model.train()

        # Test forward pass
        x = torch.randn(*input_shape, device=device)
        y = model(x)

        # Test backward pass
        if isinstance(y, dict):
            loss = y['out'].sum() if 'out' in y else list(y.values())[0].sum()
        else:
            loss = y.sum()
        loss.backward()

        return True, "✅ Architecture works on MPS"

    except NotImplementedError as e:
        return False, f"❌ Operation not supported: {str(e)}"
    except RuntimeError as e:
        return False, f"❌ Runtime error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# Usage example:
success, msg = test_mps_compatibility(
    lambda: torchvision.models.resnet50(weights="DEFAULT")
)
print(msg)
```

### Before Starting Any Project

**Decision tree:**

```
What task do you need?
├─ Image Classification → ResNet/EfficientNet/ViT (all work)
├─ Semantic Segmentation → DeepLabV3/U-Net/FCN (all work)
├─ Object Detection
│   ├─ Planning to use RCNN? → Switch to YOLO
│   └─ Planning to use YOLO? → Works perfectly
├─ Instance Segmentation
│   ├─ Planning to use Mask RCNN? → Switch to YOLACT
│   └─ Planning to use YOLACT? → Works
└─ Custom architecture → Test with test_mps_compatibility() first
```

---

## Part 5: Common Errors and Solutions

### Error: `NotImplementedError: The operator 'torchvision::roi_align' is not currently implemented`

**Cause:** Using RCNN-based model  
**Solution:** Switch to YOLO, EfficientDet, or DETR

### Error: Training runs but very slow (hours per epoch)

**Diagnosis:**

```python
# Check if tensors are actually on MPS
print(next(model.parameters()).device)  # Should show: mps:0

# Disable CPU fallback to catch issues
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
```

**Solution:**

- Update to latest PyTorch (2.0+)
- Verify no unsupported operations in model
- Use test_mps_compatibility() to isolate issue

### Error: `MPS backend out of memory`

**Cause:** Batch size too large  
**Solution:**

- Reduce batch_size
- Use gradient accumulation:

```python
accumulation_steps = 4
for i, (x, y) in enumerate(loader):
    loss = model(x, y) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Error: `RuntimeError: Placeholder storage has not been allocated on MPS device`

**Cause:** PyTorch MPS bug in versions < 2.1  
**Solution:** Update to PyTorch 2.1+

### DataLoader hangs or crashes

**Cause:** `num_workers > 0` on Mac  
**Solution:** Always use `num_workers=0`:

```python
loader = DataLoader(dataset, batch_size=32, num_workers=0)
```

---

## Part 6: Performance Profiling

### Always Profile Before Full Training

```python
import time

# Test data loading speed
t0 = time.time()
for i, batch in enumerate(train_loader):
    if i >= 10: break
dt = time.time() - t0
print(f"Data loading: {dt/10:.3f}s per batch")

# Test single training step
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

t0 = time.time()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()
dt = time.time() - t0
print(f"Single train step: {dt:.3f}s")

# Estimate full epoch time
steps_per_epoch = len(train_loader)
estimated_minutes = (dt * steps_per_epoch) / 60
print(f"Estimated time per epoch: {estimated_minutes:.1f} minutes")

# Sanity check
if estimated_minutes > 30:
    print("⚠️  WARNING: Training seems too slow. Check if MPS is being used.")
```

### Monitor GPU Usage

```bash
# Check GPU memory and utilization
sudo powermetrics --samplers gpu_power -i1000 -n1
```

---

## Part 7: Best Practices

### Batch Size Guidelines

With 96GB unified memory, you can use larger batches than typical GPUs:

- Image classification (224×224): batch_size = 64-128
- Semantic segmentation (384×384): batch_size = 16-32
- Object detection: batch_size = 8-16

**Start conservative, increase gradually.**

### Mixed Precision (Use with Caution)

```python
# MPS supports float16, but may be slower
# Test before using in production
scaler = torch.cuda.amp.GradScaler()  # Works on MPS too
with torch.cuda.amp.autocast():  # Works on MPS
    output = model(x)
    loss = criterion(output, y)
```

### Memory Management

```python
# Clear cache if running multiple experiments
import gc
gc.collect()
torch.mps.empty_cache()  # If available in your PyTorch version
```

---

## Part 8: Architecture Selection Priority

### Image Classification

1. **EfficientNet** (best accuracy/speed balance)
2. **Vision Transformer** (best for large datasets)
3. **ResNet** (reliable baseline)

### Semantic Segmentation

1. **DeepLabV3+** (best overall performance)
2. **U-Net** (best for medical imaging)
3. **SegFormer** (transformer-based)

### Object Detection

1. **YOLOv8** (fastest, good accuracy)
2. **YOLOv11** (newest version)
3. **EfficientDet** (higher precision)
4. **DETR** (transformer-based, slower)

### Instance Segmentation

1. **YOLACT** (works on MPS)
2. **YOLO + segmentation head** (custom solution)

---

## Part 9: When to Use MLX vs MPS

**MPS (PyTorch):**

- ✅ Use for training
- ✅ Better ecosystem (torchvision, transformers, etc.)
- ✅ Compatible with existing PyTorch code
- ❌ Some operations not supported

**MLX (Apple's framework):**

- ✅ Better optimized for Apple Silicon
- ✅ Faster inference
- ✅ Lower memory usage
- ❌ Different API (not PyTorch-compatible)
- ❌ Smaller ecosystem
- ❌ Less mature for training

**Recommendation:** Use MPS/PyTorch for all training. Only consider MLX for deployment/inference.

---

## Part 10: Maintaining This Document

**After each project, add notes:**

```markdown
### [Date] - [Project Type]

Task: [Classification/Detection/Segmentation]
Architecture: [Model used]
Result: [✅ Worked / ❌ Failed]
Performance: [X min/epoch, batch_size=Y]
Issues: [Any problems encountered]
Solution: [How you fixed it]
```

**Example:**

```markdown
### 2025-11-08 - Object Detection

Task: Object Detection
Architecture: YOLOv8
Result: ✅ Worked perfectly
Performance: 2.1 min/epoch, batch_size=16
Issues: Initially tried Faster RCNN, got ROI operation error
Solution: Switched to YOLO, trained successfully
```

---

## Resources

**Official Documentation:**

- PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html
- Apple Developer: https://developer.apple.com/metal/pytorch/
- MPS Op Coverage: https://github.com/pytorch/pytorch/issues/77764

**Communities:**

- PyTorch Forums: https://discuss.pytorch.org/c/metal-performance-shader/38
- GitHub Issues: https://github.com/pytorch/pytorch/labels/module%3A%20mps

**Operator Status:**

- Latest MPS ops: https://github.com/pytorch/pytorch/issues/141287

---

**Last Updated:** [Update with each project]  
**PyTorch Version:** [Current version]  
**Status:** Living document - update continuously
