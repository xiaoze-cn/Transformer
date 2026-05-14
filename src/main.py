from pathlib import Path
import argparse
import math
import random

import matplotlib.pyplot as plt
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import cfg


PAD = 0
BOS = 1
EOS = 2


def device():
    name = cfg.train.device
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def preprocess():
    data = cfg.data
    Path(data.data_dir).mkdir(parents=True, exist_ok=True)
    Path(data.raw_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        data.dataset_name,
        data.dataset_config,
        cache_dir=data.raw_dir,
        trust_remote_code=True,
    )
    splits = {"train": "train", "validation": "valid", "test": "test"}
    tok_path = Path(data.data_dir) / "tokenizer_train.txt"

    with tok_path.open("w", encoding="utf-8") as tok_f:
        for hf_name, out_name in splits.items():
            src_file = Path(data.data_dir) / f"{out_name}.{data.source_lang}"
            tgt_file = Path(data.data_dir) / f"{out_name}.{data.target_lang}"
            count = 0
            with src_file.open("w", encoding="utf-8") as src_f, tgt_file.open("w", encoding="utf-8") as tgt_f:
                for row in ds[hf_name]:
                    pair = row["translation"]
                    src = " ".join(pair[data.source_lang].strip().split())
                    tgt = " ".join(pair[data.target_lang].strip().split())
                    if not src or not tgt:
                        continue
                    src_f.write(src + "\n")
                    tgt_f.write(tgt + "\n")
                    if out_name == "train":
                        tok_f.write(src + "\n" + tgt + "\n")
                    count += 1
            print(f"{out_name}: {count} sentence pairs")

    spm.SentencePieceTrainer.train(
        input=str(tok_path),
        model_prefix=data.tokenizer_prefix,
        vocab_size=data.vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=PAD,
        bos_id=BOS,
        eos_id=EOS,
        unk_id=3,
    )


class TranslationData(Dataset):
    def __init__(self, split, sp):
        data = cfg.data
        src_path = Path(data.data_dir) / f"{split}.{data.source_lang}"
        tgt_path = Path(data.data_dir) / f"{split}.{data.target_lang}"
        max_len = data.max_length
        self.items = []
        with src_path.open("r", encoding="utf-8") as src_f, tgt_path.open("r", encoding="utf-8") as tgt_f:
            for src, tgt in zip(src_f, tgt_f):
                src_ids = [BOS] + sp.encode(src.strip(), out_type=int)[: max_len - 2] + [EOS]
                tgt_ids = [BOS] + sp.encode(tgt.strip(), out_type=int)[: max_len - 2] + [EOS]
                self.items.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def batchify(batch):
    srcs, tgts = zip(*batch)
    src = torch.full((len(batch), max(map(len, srcs))), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), max(map(len, tgts))), PAD, dtype=torch.long)
    for i, (s, t) in enumerate(batch):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class Translator(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        model_cfg = cfg.model
        dim = model_cfg.d_model
        self.dim = dim
        self.src_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = PositionalEncoding(dim, model_cfg.dropout)
        self.net = nn.Transformer(
            d_model=dim,
            nhead=model_cfg.nhead,
            num_encoder_layers=model_cfg.num_layers,
            num_decoder_layers=model_cfg.num_layers,
            dim_feedforward=model_cfg.dim_feedforward,
            dropout=model_cfg.dropout,
            batch_first=True,
        )
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, src, tgt):
        src_pad = src.eq(PAD)
        tgt_pad = tgt.eq(PAD)
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device, dtype=torch.bool), diagonal=1)
        src = self.pos(self.src_emb(src) * math.sqrt(self.dim))
        tgt = self.pos(self.tgt_emb(tgt) * math.sqrt(self.dim))
        x = self.net(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(x)


def load_sp():
    return spm.SentencePieceProcessor(model_file=f"{cfg.data.tokenizer_prefix}.model")


def train_epoch(model, loader, loss_fn, optim, dev):
    model.train()
    total_loss = total_tokens = 0
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(dev), tgt.to(dev)
        pred = model(src, tgt[:, :-1])
        gold = tgt[:, 1:]
        loss = loss_fn(pred.reshape(-1, pred.size(-1)), gold.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optim.step()
        tokens = gold.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens


@torch.no_grad()
def valid_epoch(model, loader, loss_fn, dev):
    model.eval()
    total_loss = total_tokens = 0
    for src, tgt in loader:
        src, tgt = src.to(dev), tgt.to(dev)
        pred = model(src, tgt[:, :-1])
        gold = tgt[:, 1:]
        loss = loss_fn(pred.reshape(-1, pred.size(-1)), gold.reshape(-1))
        tokens = gold.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens


def train():
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = device()
    sp = load_sp()
    model = Translator(sp.get_piece_size()).to(dev)
    train_loader = DataLoader(TranslationData("train", sp), batch_size=cfg.train.batch_size, shuffle=True, collate_fn=batchify)
    valid_loader = DataLoader(TranslationData("valid", sp), batch_size=cfg.train.batch_size, collate_fn=batchify)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=cfg.train.label_smoothing)

    Path("checkpoints").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    history = {"train": [], "valid": []}
    best = float("inf")
    print(f"Using device: {dev}")

    for epoch in range(1, cfg.train.epochs + 1):
        tr = train_epoch(model, train_loader, loss_fn, optim, dev)
        va = valid_epoch(model, valid_loader, loss_fn, dev)
        history["train"].append(tr)
        history["valid"].append(va)
        print(f"epoch {epoch:02d}: train={tr:.4f}, valid={va:.4f}")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "vocab_size": sp.get_piece_size()}, "checkpoints/model.pt")

    plt.plot(history["train"], label="train")
    plt.plot(history["valid"], label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/loss.png", dpi=160)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Translator(ckpt["vocab_size"])
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def translate_sentence(model, sp, text, dev):
    ids = [BOS] + sp.encode(text.strip(), out_type=int)[: cfg.data.max_length - 2] + [EOS]
    src = torch.tensor([ids], device=dev)
    tgt = torch.tensor([[BOS]], device=dev)
    model.eval()
    for _ in range(cfg.decode.max_length - 1):
        next_id = int(model(src, tgt)[:, -1].argmax(-1).item())
        tgt = torch.cat([tgt, torch.tensor([[next_id]], device=dev)], dim=1)
        if next_id == EOS:
            break
    return sp.decode([i for i in tgt[0].tolist() if i not in (PAD, BOS, EOS)])


def evaluate(ckpt, limit=None):
    dev = device()
    sp = load_sp()
    model = load_checkpoint(ckpt).to(dev)
    src_path = Path(cfg.data.data_dir) / f"test.{cfg.data.source_lang}"
    tgt_path = Path(cfg.data.data_dir) / f"test.{cfg.data.target_lang}"
    preds, refs, examples = [], [], []

    with src_path.open("r", encoding="utf-8") as src_f, tgt_path.open("r", encoding="utf-8") as tgt_f:
        for i, (src, ref) in enumerate(zip(src_f, tgt_f)):
            if limit is not None and i >= limit:
                break
            src, ref = src.strip(), ref.strip()
            pred = translate_sentence(model, sp, src, dev)
            preds.append(pred)
            refs.append(ref)
            if len(examples) < 20:
                examples.append((src, ref, pred))

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/examples.txt", "w", encoding="utf-8") as f:
        f.write(f"BLEU = {bleu:.2f}\n")
        f.write(f"Test sentences = {len(preds)}\n\n")
        f.write("Translation examples:\n\n")
        for src, ref, pred in examples:
            f.write(f"Source: {src}\nReference: {ref}\nPrediction: {pred}\n\n")
    print(f"BLEU = {bleu:.2f}")
    print(f"Test sentences = {len(preds)}")
    print()
    for src, ref, pred in examples:
        print(f"Source: {src}")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preprocess", "train", "evaluate", "translate"])
    parser.add_argument("--ckpt", default="checkpoints/model.pt")
    parser.add_argument("--text")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess()
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate(args.ckpt, args.limit)
    elif args.command == "translate":
        if not args.text:
            raise SystemExit("--text is required")
        dev = device()
        print(translate_sentence(load_checkpoint(args.ckpt).to(dev), load_sp(), args.text, dev))


if __name__ == "__main__":
    main()
