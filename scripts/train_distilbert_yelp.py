"""
SentimentIQ – DistilBERT on Strict 1★/3★/5★ Yelp Dataset
==========================================================
Expected accuracy: 92–95% (transformer on clean 3-class data)

Optimizations for speed on CPU:
  - Freeze transformer layers 0-3, train layers 4-5 + head only
  - Batch size 16, max_len 128
  - Only 2 epochs (clean data converges fast)
  - Weighted loss for neutral class
  - Best checkpoint by macro F1

Time: ~35-50 min on CPU for 45k samples × 2 epochs
"""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)

import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────
SST2_MODEL  = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 2          # 2 epochs enough on clean data
LR_HEAD     = 2e-3       # higher LR for new head
LR_ENC      = 5e-5       # fine-tune last 2 transformer layers
NUM_LABELS  = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH   = os.path.join(_ROOT, "models", "distilbert_sst3")
LABEL_NAMES = {0:"Negative", 1:"Neutral", 2:"Positive"}

print("=" * 60)
print("SentimentIQ – DistilBERT Fine-Tuning")
print(f"Device: {DEVICE} | Base: {SST2_MODEL}")
print("=" * 60)

# ── Data ────────────────────────────────────────────────────────
train_df = pd.read_csv("dataset/raw/train.csv").dropna(subset=["text","label"]).reset_index(drop=True)
test_df  = pd.read_csv("dataset/raw/test.csv").dropna(subset=["text","label"]).reset_index(drop=True)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print("Distribution:", dict(train_df["label"].astype(int).value_counts().sort_index()))

class SentDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts  = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]
        self.tok    = tok
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], max_length=MAX_LEN,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}

# ── Model ────────────────────────────────────────────────────────
class DistilBert3Class(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.distilbert     = base
        self.pre_classifier = nn.Linear(768, 768)
        self.classifier     = nn.Linear(768, NUM_LABELS)
        self.dropout        = nn.Dropout(0.3)
        self.relu           = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        out    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state[:, 0]
        hidden = self.pre_classifier(hidden)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        return {"logits": self.classifier(hidden)}

# ── Load pretrained ──────────────────────────────────────────────
print("\nLoading pretrained tokenizer + encoder...")
t0 = time.time()
tokenizer  = DistilBertTokenizerFast.from_pretrained(SST2_MODEL)
base_model = DistilBertModel.from_pretrained(SST2_MODEL)

# Freeze layers 0-3, train layers 4-5 + head
for name, param in base_model.named_parameters():
    param.requires_grad = ("transformer.layer.4" in name or
                           "transformer.layer.5" in name)

model = DistilBert3Class(base_model).to(DEVICE)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Loaded in {time.time()-t0:.1f}s | Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

# ── DataLoaders ──────────────────────────────────────────────────
print("\nTokenizing datasets...")
t1 = time.time()
train_ds = SentDS(train_df["text"], train_df["label"], tokenizer)
test_ds  = SentDS(test_df["text"],  test_df["label"],  tokenizer)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
test_dl  = DataLoader(test_ds,  batch_size=64,         shuffle=False, num_workers=0, pin_memory=False)
print(f"Tokenization ready in {time.time()-t1:.1f}s | {len(train_dl)} train steps/epoch")

# ── Optimizer ────────────────────────────────────────────────────
head_params = (list(model.pre_classifier.parameters()) +
               list(model.classifier.parameters()))
enc_params  = [p for p in model.distilbert.parameters() if p.requires_grad]
optimizer = AdamW([{"params": head_params, "lr": LR_HEAD},
                   {"params": enc_params,  "lr": LR_ENC}], weight_decay=0.01)

total_steps  = len(train_dl) * EPOCHS
warmup_steps = max(1, total_steps // 10)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# Weighted loss: neutral gets 2× weight (still well-balanced but boost F1)
weights = torch.tensor([1.0, 2.0, 1.0], device=DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=weights)

# ── Train ────────────────────────────────────────────────────────
print(f"\nTraining {EPOCHS} epochs × {len(train_dl)} steps = {total_steps} total steps...")
best_macro_f1 = 0.0
best_state    = None

def evaluate_model():
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(ids, mask)
            preds = torch.argmax(out["logits"], dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
    acc = accuracy_score(all_labels, all_preds)
    mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    nf1 = f1_score(all_labels, all_preds, labels=[1], average="macro", zero_division=0)
    return acc, mf1, nf1, all_preds, all_labels

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dl):
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        out  = model(ids, mask)
        loss = loss_fn(out["logits"], lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item()
        if (step + 1) % 200 == 0:
            avg = total_loss / (step + 1)
            elapsed = time.time() - t0
            print(f"  Ep{epoch} Step {step+1}/{len(train_dl)} | Loss={avg:.4f} | {elapsed:.0f}s elapsed")

    avg_loss = total_loss / len(train_dl)
    acc, mf1, nf1, _, _ = evaluate_model()
    elapsed = time.time() - t0
    print(f"\nEpoch {epoch} | Loss={avg_loss:.4f} | Acc={acc*100:.2f}% | MacroF1={mf1:.4f} | NeuF1={nf1:.4f} | {elapsed:.0f}s")
    if mf1 > best_macro_f1:
        best_macro_f1 = mf1
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ⭐ New best! MacroF1={mf1:.4f}")

# ── Final eval ───────────────────────────────────────────────────
print("\nLoading best checkpoint for final evaluation...")
if best_state:
    model.load_state_dict(best_state)

acc, mf1, nf1, all_preds, all_labels = evaluate_model()
all_probs = []
model.eval()
with torch.no_grad():
    for batch in test_dl:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        out  = model(ids, mask)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
        all_probs.extend(probs)

y_pred = np.array(all_preds); y_true = np.array(all_labels)
wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
prec, rec, f1s, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], average=None, zero_division=0)
cm   = confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist()

print("\n=== FINAL RESULTS ===")
print(classification_report(y_true, y_pred, target_names=["Negative","Neutral","Positive"], zero_division=0))
print(f"Accuracy: {acc*100:.2f}%  |  Macro F1: {mf1:.4f}")

# ── Save model ───────────────────────────────────────────────────
os.makedirs(SAVE_PATH, exist_ok=True)
torch.save({"model_state_dict": model.state_dict(),
            "config": {"max_len": MAX_LEN, "num_labels": NUM_LABELS, "base_model": SST2_MODEL}},
           os.path.join(SAVE_PATH, "model.pt"))
tokenizer.save_pretrained(SAVE_PATH)
with open(os.path.join(SAVE_PATH, "config.json"), "w") as f:
    json.dump({"base_model": SST2_MODEL, "num_labels": NUM_LABELS,
               "max_len": MAX_LEN, "accuracy": float(acc), "macro_f1": float(mf1)}, f)
print(f"✅ Model saved → {SAVE_PATH}")

# ── Save metrics ─────────────────────────────────────────────────
metrics = {
    "model_name": "DistilBERT (fine-tuned on Yelp 1★/3★/5★)",
    "accuracy":    round(float(acc), 4),
    "macro_f1":    round(float(mf1), 4),
    "weighted_f1": round(float(wf1), 4),
    "per_class": {
        "Negative": {"precision":round(float(prec[0]),4),"recall":round(float(rec[0]),4),"f1":round(float(f1s[0]),4)},
        "Neutral":  {"precision":round(float(prec[1]),4),"recall":round(float(rec[1]),4),"f1":round(float(f1s[1]),4)},
        "Positive": {"precision":round(float(prec[2]),4),"recall":round(float(rec[2]),4),"f1":round(float(f1s[2]),4)},
    },
    "confusion_matrix": cm,
    "n_train": int(len(train_df)), "n_test": int(len(test_df)),
    "train_time_sec": round(time.time()-t0, 1),
    "timestamp": datetime.utcnow().isoformat(),
    "classification_report": classification_report(y_true, y_pred,
        target_names=["Negative","Neutral","Positive"], output_dict=True, zero_division=0),
}
with open("models/evaluation_results.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✅ Metrics saved → models/evaluation_results.json")

total_time = time.time() - t0
print(f"""
{'='*60}
✅ DISTILBERT TRAINING COMPLETE in {total_time/60:.1f} min
   Model    : DistilBERT (Yelp 1★/3★/5★)
   Accuracy : {acc*100:.2f}%
   Macro F1 : {mf1:.4f}
   Neutral F1: {f1s[1]:.4f}
{'='*60}
Restart Flask to pick up DistilBERT: predict.py auto-detects it.
""")
