# Troubleshooting Guide

## Common Issues and Solutions

### 1. Model Stuck in "Starting" State

**Symptoms:**
- Model shows "Starting..." indefinitely
- No logs appear in Replicate dashboard
- Predictions never complete

**Causes & Solutions:**

#### A. Setup() Timeout
- **Cause**: Model loading takes too long (>60s default timeout)
- **Solution**: 
  - Weights are pre-downloaded during `cog build`
  - Warmup run in setup() is minimal (512x512, 1 step)
  - Check build logs to ensure weights downloaded successfully

#### B. Missing Logs
- **Cause**: Python output buffering
- **Solution**: 
  - Added `sys.stdout.reconfigure(line_buffering=True)`
  - Set `PYTHONUNBUFFERED=1` in cog.yaml
  - All logs use `flush=True`

#### C. Silent Crashes
- **Cause**: Unhandled exceptions
- **Solution**:
  - Wrapped setup() and predict() in try-except
  - Full traceback printed on errors
  - Detailed logging at each step

### 2. Out of Memory (OOM)

**Symptoms:**
- Model crashes during loading
- CUDA out of memory errors

**Solutions:**
- Use `torch.bfloat16` (half precision)
- Enable `attention_slicing()`
- Clear CUDA cache after warmup
- Monitor GPU memory in logs

### 3. Model Loading Fails

**Symptoms:**
- "Model cache not found" error
- FileNotFoundError during setup

**Solutions:**
- Verify `download_weights.py` ran during build
- Check build logs for download errors
- Ensure HuggingFace Hub is accessible
- Model size: ~12GB, ensure enough disk space

### 4. Slow Inference

**Symptoms:**
- Generation takes >10 seconds
- Slower than expected

**Solutions:**
- Flash Attention enabled (if available)
- Model compilation disabled (causes first-run delay)
- Warmup run in setup() for faster first prediction
- Use default 8 steps for Z-Image-Turbo

## Debugging Checklist

Before deploying to Replicate:

- [ ] Test locally with `cog predict`
- [ ] Check build logs for errors
- [ ] Verify model weights downloaded (~12GB)
- [ ] Confirm GPU is detected in logs
- [ ] Run warmup successfully
- [ ] Test with various prompts and sizes

## Local Testing

```bash
# Build the image
cog build

# Test prediction
cog predict -i prompt="test image"

# Test with all parameters
cog predict \
  -i prompt="A beautiful landscape" \
  -i width=1024 \
  -i height=1024 \
  -i num_inference_steps=8 \
  -i seed=42
```

## Reading Logs

### Setup Phase Logs
```
[timestamp] Starting Z-Image-Turbo setup...
[timestamp] PyTorch version: 2.x.x
[timestamp] CUDA available: True
[timestamp] GPU: NVIDIA A100
[timestamp] Model cache found at: ./model-cache
[timestamp] Loading Z-Image-Turbo pipeline...
[timestamp] Pipeline loaded in X.XXs
[timestamp] Moving model to CUDA...
[timestamp] ✓ Flash Attention enabled
[timestamp] Running warmup inference...
[timestamp] ✓ Setup completed successfully!
```

### Prediction Phase Logs
```
[timestamp] Starting prediction...
[timestamp] Prompt: ...
[timestamp] Size: 1024x1024
[timestamp] Generating image...
[timestamp] ✓ Image generated in X.XXs
[timestamp] ✓ Prediction completed successfully!
```

## Key Improvements

1. **Unbuffered Output**: Real-time logs visible immediately
2. **Timestamped Logs**: Track timing of each operation
3. **Error Handling**: Full tracebacks on failures
4. **Progress Tracking**: Log every major step
5. **Resource Monitoring**: GPU memory usage logged
6. **Warmup Run**: Faster first prediction
7. **Pre-downloaded Weights**: No download during runtime

## If Still Stuck

1. Check Replicate build logs first
2. Verify model pushed successfully: `cog push`
3. Test locally before deploying
4. Check Replicate status page for platform issues
5. Contact Replicate support with logs

## Performance Expectations

- **Build time**: 10-15 minutes (downloading 12GB model)
- **Setup time**: 30-60 seconds (loading model to GPU)
- **First prediction**: 5-10 seconds (includes warmup)
- **Subsequent predictions**: 2-5 seconds (1024x1024, 8 steps)
