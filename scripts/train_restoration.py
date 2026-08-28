"""
train_restoration.py — train the Ge'ez/Amharic restoration transformer.

Loss is computed only at damaged positions. Scoring every position would let the
model bank easy accuracy by copying visible characters, which flatters the
metric and teaches nothing about restoration.

Damage is generated fresh each epoch rather than baked into a fixed dataset, so
a given phrase is seen under many different loss patterns.

Baseline to beat: the character n-gram in scripts/run_corpus_scaling.py reaches
27.73% top-1 on Ge'ez (AGE, 1.15M characters).

Usage:
    python scripts/train_restoration.py --corpus data/restoration_corpus_geez_age.json \\
        --language geez --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ocr.damage import build_training_pair
from src.ocr.restoration_model import (
    NUM_VOWEL_CLASSES,
    PAD_TOKEN,
    VOWEL_NONE,
    FidelVocab,
    GeezRestorationModel,
    RestorationConfig,
    save_restoration_model,
)

IGNORE_INDEX = -100


class LengthBucketSampler(torch.utils.data.Sampler):
    """
    Group similar-length phrases into batches.

    Attention is quadratic in sequence length, so a batch padded to its longest
    member wastes work on every shorter one. Bucketing keeps batches uniform;
    batch order is still shuffled so training does not see length as a schedule.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool = True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [order[i:i + self.batch_size]
                   for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


class RestorationDataset(Dataset):
    """Clean phrases damaged on the fly."""

    def __init__(self, phrases: list[str], vocab: FidelVocab, max_len: int,
                 fixed_seed: int | None = None, damage_rate: float | None = None):
        self.phrases = phrases
        self.vocab = vocab
        self.max_len = max_len
        self.fixed_seed = fixed_seed
        self.damage_rate = damage_rate

    def __len__(self) -> int:
        return len(self.phrases)

    def __getitem__(self, index: int):
        clean = self.phrases[index]
        seed = None if self.fixed_seed is None else self.fixed_seed + index
        damaged, targets = build_training_pair(clean, self.damage_rate, seed=seed)

        dmg_base, dmg_vowel = self.vocab.encode_tokens(damaged)
        target_tokens = [t if t is not None else PAD_TOKEN for t in targets]
        tgt_base, tgt_vowel = self.vocab.encode_tokens(target_tokens)

        base_target = [b if t is not None else IGNORE_INDEX
                       for b, t in zip(tgt_base, targets)]
        vowel_target = [v if t is not None else IGNORE_INDEX
                        for v, t in zip(tgt_vowel, targets)]

        return (dmg_base[:self.max_len], dmg_vowel[:self.max_len],
                base_target[:self.max_len], vowel_target[:self.max_len])


def collate(batch, pad_idx: int):
    """Pad a batch to its longest sequence."""
    width = max(len(item[0]) for item in batch)
    bases, vowels, base_t, vowel_t, masks = [], [], [], [], []
    for b, v, bt, vt in batch:
        pad = width - len(b)
        bases.append(b + [pad_idx] * pad)
        vowels.append(v + [VOWEL_NONE] * pad)
        base_t.append(bt + [IGNORE_INDEX] * pad)
        vowel_t.append(vt + [IGNORE_INDEX] * pad)
        masks.append([False] * len(b) + [True] * pad)
    return (torch.tensor(bases), torch.tensor(vowels), torch.tensor(base_t),
            torch.tensor(vowel_t), torch.tensor(masks))


def load_phrases(corpus: Path, max_len: int) -> list[str]:
    """Read chunked corpus JSON and drop anything too long."""
    payload = json.loads(corpus.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("chunks", [])
    phrases = []
    for item in items:
        text = item if isinstance(item, str) else item.get("text", "")
        text = " ".join(text.split())
        if 8 <= len(text) <= max_len:
            phrases.append(text)
    return phrases


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False) -> tuple[float, float]:
    """Top-1 accuracy over damaged positions: whole fidel, and base only."""
    model.eval()
    exact = base_ok = total = 0
    for bases, vowels, base_t, vowel_t, mask in loader:
        bases, vowels = bases.to(device), vowels.to(device)
        base_t, vowel_t, mask = base_t.to(device), vowel_t.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            base_logits, vowel_logits = model(bases, vowels, mask)
        scored = base_t != IGNORE_INDEX
        if not scored.any():
            continue
        pb = base_logits.argmax(-1)[scored]
        pv = vowel_logits.argmax(-1)[scored]
        tb = base_t[scored]
        tv = vowel_t[scored]
        base_ok += (pb == tb).sum().item()
        exact += ((pb == tb) & (pv == tv)).sum().item()
        total += scored.sum().item()
    return (exact / max(total, 1), base_ok / max(total, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-damage-rate", type=float, default=0.25,
                        help="Fixed rate for validation. Matches the n-gram "
                             "baseline so the two are measured on one task; "
                             "training still samples the full 0-0.75 range.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Dataloader workers; damage is generated per sample "
                             "so this is CPU-bound and worth raising on GPU")
    parser.add_argument("--amp", action="store_true",
                        help="bfloat16 autocast; roughly halves step time on Ada")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seeds the corpus shuffle only. Kept separate from "
                             "--seed so a seed sweep varies initialisation and "
                             "damage sampling while the train/val split - and "
                             "therefore the vocabulary, which is built from the "
                             "training half - stays fixed.")
    parser.add_argument("--tag", default="",
                        help="Suffix for the model and log filenames. A sweep "
                             "otherwise overwrites its own results.")
    parser.add_argument("--limit", type=int, default=0, help="0 = all phrases")
    parser.add_argument("--limit-chars", type=int, default=0,
                        help="Cap training text by character count. Chunk lengths "
                             "differ between corpora (Ge'ez AGE averages 64.9, "
                             "Amharic 32.5), so equal phrase counts are not equal "
                             "amounts of text; use this to compare languages.")
    parser.add_argument("--out", type=Path,
                        default=Path("models/restoration"))
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the per-epoch checkpoint if one exists. "
                             "Kaggle wipes /kaggle/working when a session restarts, "
                             "so an uncommitted run is otherwise lost entirely.")
    args = parser.parse_args()

    run_name = f"{args.language}_{args.tag}" if args.tag else args.language

    phrases = load_phrases(args.corpus, args.max_len)
    # Shuffle first: corpus order is canonical book order, so slicing the head
    # would sample Genesis rather than the whole work.
    random.Random(args.split_seed).shuffle(phrases)
    if args.limit:
        phrases = phrases[:args.limit]
    if args.limit_chars:
        kept, used = [], 0
        for phrase in phrases:
            if used >= args.limit_chars:
                break
            kept.append(phrase)
            used += len(phrase)
        phrases = kept
    if len(phrases) < 50:
        print(f"Only {len(phrases)} usable phrases; need more", file=sys.stderr)
        return 1

    split = max(1, int(len(phrases) * args.val_fraction))
    val_phrases, train_phrases = phrases[:split], phrases[split:]
    vocab = FidelVocab.from_corpus(train_phrases)

    # Only now, so everything above depends on --split-seed alone.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"{args.language}: {len(train_phrases):,} train | {len(val_phrases):,} val "
          f"| {sum(len(p) for p in train_phrases):,} train chars "
          f"| vocab {len(vocab)} | eval damage {args.eval_damage_rate:.0%} "
          f"| seed {args.seed} | split-seed {args.split_seed}")

    train_set = RestorationDataset(train_phrases, vocab, args.max_len)
    val_set = RestorationDataset(val_phrases, vocab, args.max_len, fixed_seed=1234,
                                 damage_rate=args.eval_damage_rate)
    # A lambda cannot be pickled, which breaks worker processes on Windows.
    collate_fn = partial(collate, pad_idx=vocab.pad_idx)
    train_loader = DataLoader(
        train_set,
        batch_sampler=LengthBucketSampler([len(p) for p in train_phrases],
                                          args.batch_size),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=LengthBucketSampler([len(p) for p in val_phrases],
                                          args.batch_size, shuffle=False),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    config = RestorationConfig(vocab_size=len(vocab), emb_dim=args.emb_dim,
                               num_layers=args.layers, num_heads=args.heads,
                               mlp_dim=args.emb_dim * 4, max_len=args.max_len)
    model = GeezRestorationModel(config).to(device)
    print(f"parameters: {model.parameter_count():,}  device: {device}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.epochs * max(len(train_loader), 1),
        pct_start=0.1,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    args.out.mkdir(parents=True, exist_ok=True)
    resume_path = args.out / f"{run_name}_last.pth"
    best = 0.0
    history = []
    start_epoch = 1

    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("vocab_fingerprint") != vocab.fingerprint():
            print("Checkpoint vocabulary does not match this corpus; starting fresh.",
                  file=sys.stderr)
        else:
            model.load_state_dict(state["state_dict"])
            optimiser.load_state_dict(state["optimiser"])
            # OneCycle bakes total_steps into its state, so restoring it after
            # --epochs changed would overrun the schedule. Rebuild and
            # fast-forward in that case.
            expected_steps = args.epochs * max(len(train_loader), 1)
            if state["scheduler"].get("total_steps") == expected_steps:
                scheduler.load_state_dict(state["scheduler"])
            else:
                print(f"--epochs changed since the checkpoint; rebuilding schedule "
                      f"for {args.epochs} epochs")
                for _ in range(min(state["epoch"] * len(train_loader),
                                   expected_steps - 1)):
                    scheduler.step()
            best = state["best"]
            history = state["history"]
            start_epoch = state["epoch"] + 1
            print(f"resumed from epoch {state['epoch']} (best {best:.2%})")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for bases, vowels, base_t, vowel_t, mask in train_loader:
            bases, vowels = bases.to(device), vowels.to(device)
            base_t, vowel_t, mask = base_t.to(device), vowel_t.to(device), mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                base_logits, vowel_logits = model(bases, vowels, mask)
                loss = (loss_fn(base_logits.reshape(-1, base_logits.size(-1)),
                                base_t.reshape(-1))
                        + loss_fn(vowel_logits.reshape(-1, NUM_VOWEL_CLASSES),
                                  vowel_t.reshape(-1)))

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            running += loss.item()

        exact, base_only = evaluate(model, val_loader, device, amp=args.amp)
        history.append({"epoch": epoch, "loss": running / max(len(train_loader), 1),
                        "top1": exact, "base_top1": base_only})
        print(f"  epoch {epoch:>3} | loss {running / max(len(train_loader), 1):6.3f} "
              f"| top1 {exact:6.2%} | base {base_only:6.2%} "
              f"| {time.time() - started:5.1f}s")

        if exact > best:
            best = exact
            save_restoration_model(model, vocab,
                                   args.out / f"{run_name}_restoration.pth")

        torch.save({
            "state_dict": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "history": history,
            "vocab_fingerprint": vocab.fingerprint(),
        }, resume_path)

    log_path = Path("logs/restoration") / f"{run_name}_training.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "language": args.language,
        "corpus": str(args.corpus),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "parameters": model.parameter_count(),
        "best_top1": best,
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"\nbest top-1 {best:.2%} -> {args.out / f'{run_name}_restoration.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
