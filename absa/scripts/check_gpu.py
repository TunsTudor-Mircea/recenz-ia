"""
GPU pre-flight check for ABSA training.

Validates:
  1. PyTorch CUDA availability and device properties
  2. fp16 (half-precision) tensor operations on GPU
  3. End-to-end forward + backward pass with a real BERT-family model
     at the training batch size (default 32, seq_len 128)
  4. Peak VRAM consumption and OOM safety margin

Usage
-----
    # Quick check with a tiny model (fast, ~200MB download)
    python absa/scripts/check_gpu.py --model distilbert-base-multilingual-cased

    # Full check with the actual training model (recommended before first run)
    python absa/scripts/check_gpu.py --model dumitrescustefan/bert-base-romanian-cased-v1
    python absa/scripts/check_gpu.py --model bert-base-multilingual-cased
    python absa/scripts/check_gpu.py --model xlm-roberta-base

    # Check a different batch size
    python absa/scripts/check_gpu.py --model bert-base-multilingual-cased --batch-size 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

PASS = '\033[92m PASS\033[0m'
FAIL = '\033[91m FAIL\033[0m'
WARN = '\033[93m WARN\033[0m'

NUM_LABELS = 4
MAX_LENGTH = 128


def section(title: str) -> None:
    log.info('')
    log.info('─' * 60)
    log.info('  %s', title)
    log.info('─' * 60)


def check_torch() -> 'torch':
    section('1 · PyTorch + CUDA')
    try:
        import torch
        log.info('[%s] torch imported   version=%s', 'OK', torch.__version__)
    except ImportError:
        log.error('[%s] torch not installed.  pip install torch --break-system-packages', FAIL)
        sys.exit(1)

    if not torch.cuda.is_available():
        log.error('[%s] torch.cuda.is_available() = False', FAIL)
        log.error('      PyTorch cannot see the GPU.  Check that:')
        log.error('        • CUDA toolkit is installed (CUDA 13.1 detected by nvidia-smi)')
        log.error('        • torch was installed with CUDA support:')
        log.error('            pip install torch --index-url https://download.pytorch.org/whl/cu124')
        sys.exit(1)

    log.info('[%s] CUDA available', PASS)

    n = torch.cuda.device_count()
    log.info('[%s] GPU count: %d', 'OK', n)
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        log.info('[%s] GPU %d: %s  VRAM=%.1f GB  SM=%d.%d  CUDA=%s',
                 'OK', i, props.name, vram_gb,
                 props.major, props.minor,
                 torch.version.cuda)

    return torch


def check_fp16(torch) -> None:
    section('2 · FP16 (half-precision) smoke test')
    try:
        a = torch.randn(128, 768, device='cuda', dtype=torch.float16)
        b = torch.randn(768, 128, device='cuda', dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        log.info('[%s] fp16 matrix multiply (128×768 @ 768×128) succeeded', PASS)
    except Exception as e:
        log.error('[%s] fp16 failed: %s', FAIL, e)
        sys.exit(1)


def check_transformer_forward(torch, model_name: str, batch_size: int) -> None:
    section(f'3 · Transformer forward + backward  (model={model_name}  batch={batch_size}  len={MAX_LENGTH})')

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        log.error('[%s] transformers not installed.  pip install transformers', FAIL)
        sys.exit(1)

    # ── load tokeniser ────────────────────────────────────────────────────────
    log.info('Loading tokeniser …')
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        log.info('[%s] tokeniser loaded', PASS)
    except Exception as e:
        log.error('[%s] tokeniser load failed: %s', FAIL, e)
        sys.exit(1)

    # ── load model ────────────────────────────────────────────────────────────
    log.info('Loading model (this downloads weights if not cached) …')
    t0 = time.time()
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_LABELS
        )
        model = model.half().cuda()   # fp16 on GPU — same as training
        model.train()
        log.info('[%s] model loaded in %.1fs', PASS, time.time() - t0)
    except Exception as e:
        log.error('[%s] model load failed: %s', FAIL, e)
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info('[  ] Parameters: %.1fM', n_params)

    # ── build a synthetic batch ───────────────────────────────────────────────
    text_a = ['Bateria telefonului este excelentă, ține o zi întreagă fără probleme.'] * batch_size
    text_b = ['baterie și autonomie'] * batch_size

    enc = tok(
        text_a, text_b,
        truncation=True, padding='max_length',
        max_length=MAX_LENGTH, return_tensors='pt',
    )
    input_ids      = enc['input_ids'].cuda()
    attention_mask = enc['attention_mask'].cuda()
    labels         = torch.zeros(batch_size, dtype=torch.long).cuda()

    # token_type_ids only for BERT-family
    token_type_ids = enc.get('token_type_ids')
    if token_type_ids is not None:
        token_type_ids = token_type_ids.cuda()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # ── forward pass ─────────────────────────────────────────────────────────
    log.info('Running forward pass …')
    try:
        t0 = time.time()
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids

        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(**kwargs)

        loss = out.loss
        log.info('[%s] forward pass   loss=%.4f   time=%.2fs', PASS, loss.item(), time.time() - t0)
    except torch.cuda.OutOfMemoryError:
        log.error('[%s] OOM on forward pass with batch_size=%d', FAIL, batch_size)
        log.error('      Rerun with --batch-size 16 or add --grad-accum 2')
        torch.cuda.empty_cache()
        sys.exit(1)
    except Exception as e:
        log.error('[%s] forward pass failed: %s', FAIL, e)
        sys.exit(1)

    # ── backward pass ─────────────────────────────────────────────────────────
    log.info('Running backward pass …')
    try:
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        log.info('[%s] backward pass   time=%.2fs', PASS, time.time() - t0)
    except torch.cuda.OutOfMemoryError:
        log.error('[%s] OOM on backward pass with batch_size=%d', FAIL, batch_size)
        log.error('      Rerun with --batch-size 16 or add --grad-accum 2')
        torch.cuda.empty_cache()
        sys.exit(1)
    except Exception as e:
        log.error('[%s] backward pass failed: %s', FAIL, e)
        sys.exit(1)

    # ── VRAM report ───────────────────────────────────────────────────────────
    section('4 · VRAM consumption')
    peak_bytes   = torch.cuda.max_memory_allocated()
    total_bytes  = torch.cuda.get_device_properties(0).total_memory
    peak_gb      = peak_bytes  / 1024**3
    total_gb     = total_bytes / 1024**3
    used_pct     = 100 * peak_bytes / total_bytes
    headroom_gb  = total_gb - peak_gb

    log.info('Peak VRAM allocated : %.2f GB / %.2f GB  (%.0f%%)', peak_gb, total_gb, used_pct)
    log.info('Headroom            : %.2f GB', headroom_gb)

    if used_pct < 70:
        log.info('[%s] Comfortable — batch_size=%d should be safe for training.', PASS, batch_size)
    elif used_pct < 88:
        log.info('[%s] Tight but should be OK.  Monitor with nvidia-smi dmon during training.', WARN)
        log.info('       If you hit OOM during training, add --grad-accum 2 and halve --batch-size.')
    else:
        log.error('[%s] %.0f%% VRAM on a single forward+backward.  Training will OOM.', FAIL, used_pct)
        log.error('       Run with: --batch-size %d --grad-accum 2', batch_size // 2)
        sys.exit(1)

    # ── throughput estimate ───────────────────────────────────────────────────
    section('5 · Throughput estimate')
    torch.cuda.reset_peak_memory_stats()

    # Time 5 forward passes
    repeats = 5
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    secs_per_batch = elapsed / repeats
    samples_per_sec = batch_size / secs_per_batch
    train_batches   = 124_760 // batch_size   # train set size
    epoch_mins      = (train_batches * secs_per_batch) / 60
    total_mins_5ep  = epoch_mins * 5

    log.info('Throughput      : %.0f samples/sec  (%.2fs per batch of %d)',
             samples_per_sec, secs_per_batch, batch_size)
    log.info('Est. per epoch  : %.1f min  (%d steps)',
             epoch_mins, train_batches)
    log.info('Est. 5 epochs   : %.0f min  (early stopping may cut this)',
             total_mins_5ep)

    # ── done ──────────────────────────────────────────────────────────────────
    section('Summary')
    log.info('[%s] All checks passed — ready to run train_transformer.py', PASS)
    log.info('')
    log.info('Recommended command:')
    log.info('  python absa/scripts/train_transformer.py --model %s', model_name)
    if used_pct >= 70:
        log.info('  (or with tighter batch: --batch-size 16 --grad-accum 2)')


def main() -> None:
    p = argparse.ArgumentParser(description='GPU pre-flight check for ABSA training.')
    p.add_argument('--model', default='dumitrescustefan/bert-base-romanian-cased-v1',
                   help='HuggingFace model name to test.')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Batch size to simulate (default: 32, same as training).')
    args = p.parse_args()

    torch = check_torch()
    check_fp16(torch)
    check_transformer_forward(torch, args.model, args.batch_size)


if __name__ == '__main__':
    main()
