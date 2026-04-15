"""
Fine-tune a pre-trained transformer for ABSA sentence-pair classification.

Implements the sentence-pair approach of Sun et al. (2019): "Utilizing BERT
for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence"
(NAACL 2019, arXiv:1903.09588).

For each (review, aspect) pair the tokenizer receives:
    [CLS] text_a [SEP] text_b [SEP]
where text_a is the review and text_b is the Romanian aspect auxiliary sentence.
A linear classification head on the [CLS] token predicts one of 4 classes:
    0: none  1: positive  2: negative  3: neutral

Supported base models
---------------------
    mBERT    : bert-base-multilingual-cased   (Devlin et al. 2019)
    RoBERT   : dumitrescustefan/bert-base-romanian-cased-v1  (Dumitrescu et al. 2020)
    XLM-R    : xlm-roberta-base               (Conneau et al. 2020)

Training details
----------------
    Optimizer  : AdamW (weight_decay=0.01)
    LR schedule: linear warmup (10% of steps) + linear decay
    Loss       : CrossEntropy with inverse-frequency class weights
                 (addresses "none" class dominance, ~87% of pairs)
    Epochs     : 5 (early stopping on val macro-F1, patience=2)
    Batch size : 32 (effective; use gradient accumulation if GPU < 16GB)
    Max length : 128 tokens (sufficient for title+content+aspect)
    Mixed prec.: fp16 if CUDA available

All random seeds are fixed to RANDOM_SEED = 42.

Outputs
-------
    absa/models/checkpoints/{run_name}/   HuggingFace checkpoint
    absa/models/checkpoints/{run_name}/training_args.json
    absa/models/checkpoints/{run_name}/metrics.json     val + test metrics

Usage
-----
    # Fine-tune Romanian BERT (run from repo root)
    python absa/scripts/train_transformer.py --model dumitrescustefan/bert-base-romanian-cased-v1

    # Fine-tune mBERT
    python absa/scripts/train_transformer.py --model bert-base-multilingual-cased

    # Fine-tune XLM-RoBERTa with custom run name
    python absa/scripts/train_transformer.py --model xlm-roberta-base --run-name xlmr_run1

    # Adjust batch / epochs
    python absa/scripts/train_transformer.py --model xlm-roberta-base --batch-size 16 --epochs 3

    # Resume from checkpoint
    python absa/scripts/train_transformer.py --model xlm-roberta-base --resume-from absa/models/checkpoints/xlm-roberta-base/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

SPLITS_DIR = Path('absa/data/splits')
CKPT_DIR   = Path('absa/models/checkpoints')

ASPECT_LABELS = [
    'BATERIE', 'ECRAN', 'SUNET', 'PERFORMANTA', 'CONECTIVITATE',
    'DESIGN', 'CALITATE_CONSTRUCTIE', 'PRET', 'LIVRARE', 'GENERAL',
]

LABEL2ID = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 4

# ── reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── dataset class ─────────────────────────────────────────────────────────────

class ABSADataset:
    """
    torch.utils.data.Dataset for sentence-pair ABSA classification.

    Each example:
        input_ids        : [CLS] text_a [SEP] text_b [SEP]  (padded to max_length)
        attention_mask   : 1 for real tokens, 0 for padding
        token_type_ids   : 0 for text_a tokens, 1 for text_b tokens (if supported)
        labels           : integer in {0,1,2,3}
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        self.encodings = tokenizer(
            list(df['text_a']),
            list(df['text_b']),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        self.labels = df['label_id'].values

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        import torch
        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


# ── evaluation helper ─────────────────────────────────────────────────────────

def compute_absa_metrics(eval_pred):
    """
    HuggingFace Trainer compute_metrics callback.

    Returns macro detection-F1 and macro polarity-F1, plus the primary metric
    (detection_macro_f1) used for early stopping / best-model selection.
    """
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    labels = np.array(labels)
    preds  = np.array(preds)

    # Detection: binary (none vs. any)
    det_true = (labels > 0).astype(int)
    det_pred = (preds  > 0).astype(int)
    det_f1   = f1_score(det_true, det_pred, zero_division=0)

    # Polarity: among true non-none, 3-class macro
    mask = labels > 0
    pol_f1 = (
        f1_score(labels[mask], preds[mask], average='macro',
                 labels=[1, 2, 3], zero_division=0)
        if mask.sum() >= 2 else 0.0
    )

    # Combined: 4-class macro (includes none)
    comb_f1 = f1_score(labels, preds, average='macro',
                       labels=[0, 1, 2, 3], zero_division=0)

    return {
        'detection_f1':        round(float(det_f1), 4),
        'polarity_macro_f1':   round(float(pol_f1), 4),
        'combined_macro_f1':   round(float(comb_f1), 4),
    }


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(labels: pd.Series) -> 'torch.Tensor':
    """Inverse-frequency class weights for CrossEntropyLoss."""
    import torch
    counts = np.bincount(labels.values, minlength=NUM_LABELS).astype(float)
    counts = np.maximum(counts, 1)       # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_LABELS   # normalise to sum=NUM_LABELS
    return torch.tensor(weights, dtype=torch.float)


# ── custom Trainer with class weighting ───────────────────────────────────────

def make_weighted_trainer_class(class_weights):
    """Return a Trainer subclass that uses weighted CrossEntropyLoss."""
    try:
        import torch
        from transformers import Trainer

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop('labels')
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=class_weights.to(logits.device)
                )
                loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        return WeightedTrainer

    except ImportError:
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fine-tune a transformer for ABSA sentence-pair classification.')
    p.add_argument(
        '--model', required=True,
        help='HuggingFace model name or local path. '
             'Examples: bert-base-multilingual-cased, '
             'dumitrescustefan/bert-base-romanian-cased-v1, xlm-roberta-base',
    )
    p.add_argument('--run-name',   default=None,
                   help='Run name for checkpoint folder (default: model basename).')
    p.add_argument('--epochs',     type=int,   default=5)
    p.add_argument('--batch-size', type=int,   default=32)
    p.add_argument('--grad-accum', type=int,   default=1,
                   help='Gradient accumulation steps (increase if OOM).')
    p.add_argument('--lr',         type=float, default=2e-5)
    p.add_argument('--max-length', type=int,   default=128)
    p.add_argument('--warmup-ratio', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--resume-from', default=None,
                   help='Resume training from a saved checkpoint directory.')
    p.add_argument('--no-class-weights', action='store_true',
                   help='Disable inverse-frequency class weighting.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            TrainingArguments,
        )
    except ImportError:
        log.error('transformers and torch are required:')
        log.error('  pip install transformers torch --break-system-packages')
        raise SystemExit(1)

    # ── load splits ────────────────────────────────────────────────────────────
    for path in (SPLITS_DIR / 'train.csv', SPLITS_DIR / 'val.csv', SPLITS_DIR / 'test_gold.csv'):
        if not path.exists():
            log.error('%s not found. Run prepare_training_data.py first.', path)
            raise SystemExit(1)

    log.info('Loading data splits …')
    train_df = pd.read_csv(SPLITS_DIR / 'train.csv',     dtype={'review_id': str})
    val_df   = pd.read_csv(SPLITS_DIR / 'val.csv',       dtype={'review_id': str})
    test_df  = pd.read_csv(SPLITS_DIR / 'test_gold.csv', dtype={'review_id': str})
    log.info('Train: %d  Val: %d  Test: %d', len(train_df), len(val_df), len(test_df))

    # ── tokeniser ──────────────────────────────────────────────────────────────
    run_name = args.run_name or Path(args.model).name
    log.info('Loading tokeniser: %s', args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── datasets ───────────────────────────────────────────────────────────────
    log.info('Tokenising datasets (max_length=%d) …', args.max_length)
    train_dataset = ABSADataset(train_df, tokenizer, args.max_length)
    val_dataset   = ABSADataset(val_df,   tokenizer, args.max_length)
    test_dataset  = ABSADataset(test_df,  tokenizer, args.max_length)

    # ── model ──────────────────────────────────────────────────────────────────
    log.info('Loading model: %s  (num_labels=%d)', args.model, NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── class weights ──────────────────────────────────────────────────────────
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_df['label_id'])
        log.info('Class weights: %s', dict(zip(ID2LABEL.values(), class_weights.tolist())))

    # ── training arguments ─────────────────────────────────────────────────────
    ckpt_path = CKPT_DIR / run_name
    ckpt_path.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    total_steps = (len(train_df) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    log.info('Total steps: %d  Warmup steps: %d  fp16: %s', total_steps, warmup_steps, use_fp16)

    training_args = TrainingArguments(
        output_dir=str(ckpt_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type='linear',
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='detection_f1',
        greater_is_better=True,
        fp16=use_fp16,
        seed=RANDOM_SEED,
        report_to='none',
        logging_steps=200,
        dataloader_num_workers=0,
    )

    # ── trainer ────────────────────────────────────────────────────────────────
    TrainerClass = (
        make_weighted_trainer_class(class_weights)
        if class_weights is not None else None
    )
    if TrainerClass is None:
        from transformers import Trainer as TrainerClass   # type: ignore[assignment]

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_absa_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── train ──────────────────────────────────────────────────────────────────
    resume = args.resume_from or (str(ckpt_path) if list(ckpt_path.glob('checkpoint-*')) else None)
    log.info('Starting training …  (resume_from=%s)', resume)
    trainer.train(resume_from_checkpoint=resume)

    # ── evaluate ───────────────────────────────────────────────────────────────
    log.info('Evaluating on val set …')
    val_metrics  = trainer.evaluate(val_dataset)
    log.info('Evaluating on gold test set …')
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix='test')

    log.info('Val:  %s', {k: v for k, v in val_metrics.items() if 'f1' in k})
    log.info('Test: %s', {k: v for k, v in test_metrics.items() if 'f1' in k})

    # ── save ───────────────────────────────────────────────────────────────────
    trainer.save_model(str(ckpt_path / 'best_model'))
    tokenizer.save_pretrained(str(ckpt_path / 'best_model'))

    metrics_out = {
        'model': args.model,
        'run_name': run_name,
        'val':  {k.replace('eval_', ''): v for k, v in val_metrics.items()},
        'test': {k.replace('test_', ''): v for k, v in test_metrics.items()},
    }
    with (ckpt_path / 'metrics.json').open('w') as f:
        json.dump(metrics_out, f, indent=2)

    # Save training args for reproducibility
    train_args_dict = {
        'model': args.model,
        'run_name': run_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'lr': args.lr,
        'max_length': args.max_length,
        'warmup_ratio': args.warmup_ratio,
        'weight_decay': args.weight_decay,
        'class_weights_used': not args.no_class_weights,
        'random_seed': RANDOM_SEED,
    }
    with (ckpt_path / 'training_args.json').open('w') as f:
        json.dump(train_args_dict, f, indent=2)

    log.info('Best model saved → %s/best_model', ckpt_path)
    log.info('Metrics saved → %s/metrics.json', ckpt_path)


if __name__ == '__main__':
    main()
