# 🧠 UDA + RoBERTa GPU Fine-Tuning Reference (Applied Research Setup)

## 🎯 Overview

This document defines a **complete GPU training pipeline** for **Unsupervised Data Augmentation (UDA)** using **RoBERTa**, tailored for per-topic sentiment classification on your `earningsdf_40sample_labeled.parquet` dataset.

Includes:
- Data preparation (topic-level splits)
- Anonymization
- UDA trainer implementation (RoBERTa Masked LM)
- GPU fine-tuning loop
- Evaluation and debugging guide

---

## ⚙️ Environment

```bash
pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn tqdm pandas numpy
```

GPU device assumed available via CUDA.

---

## 🧩 Step 1 — Dataset Preparation (`prepare_uda_datasets_from_parquet.py`)

```python
#!/usr/bin/env python3
import os, re, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

INPUT_PARQUET = "earningsdf_40sample_labeled.parquet"
OUTPUT_DIR = "prepared_uda_datasets"
UNLABELED_FRACTION = 0.3
SEED = 42

np.random.seed(SEED)
df = pd.read_parquet(INPUT_PARQUET)

def anonymize(name):
    if pd.isna(name): return "Company_Unknown"
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip())
    return f"Company_{abs(hash(clean)) % 10000}"

df["company_anon"] = df.get("company", "Unknown").apply(anonymize)
topics = [c.replace("sentiment_", "") for c in df.columns if c.startswith("sentiment_")]
os.makedirs(OUTPUT_DIR, exist_ok=True)

for t in tqdm(topics):
    sent, lab, conf = "sentence", f"sentiment_{t}", f"confidence_{t}"
    if lab not in df: continue
    d = df[[sent, lab, conf, "company_anon"]].rename(
        columns={lab:"ratings", conf:"confidence", "company_anon":"company"}
    ).dropna(subset=["ratings"])
    mapping = {"Positive":1, "Negative":0, "Neutral":0}
    d["ratings"] = d["ratings"].map(mapping)
    d = d.dropna(subset=["ratings"])

    if len(d) < 10: continue
    train, tmp = train_test_split(d, test_size=0.3, stratify=d["ratings"], random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.5, stratify=tmp["ratings"], random_state=SEED)
    unlabeled = d.sample(frac=UNLABELED_FRACTION, random_state=SEED)[["sentence"]]
    unlabeled["ratings"] = None

    td = os.path.join(OUTPUT_DIR, t); os.makedirs(td, exist_ok=True)
    train.to_csv(f"{td}/labeled.csv", index=False)
    val.to_csv(f"{td}/val.csv", index=False)
    test.to_csv(f"{td}/test.csv", index=False)
    unlabeled.to_csv(f"{td}/unlabeled.csv", index=False)
```

✅ Produces `prepared_uda_datasets/{topic}/` with four files:
- `labeled.csv`
- `val.csv`
- `test.csv`
- `unlabeled.csv`

---

## ⚡ Step 2 — GPU Trainer (`run_uda_training_gpu_standalone.py`)

Core idea: combine supervised loss on labeled data and unsupervised loss from pseudo-labeled augmented data.

### 🔧 Key Settings
```python
MODEL_NAME = "FacebookAI/roberta-base"
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LAMBDA = 1.0
P = 0.3  # Masking probability
```

---

### 🧠 UDA Trainer Class

```python
class UDATrainer(Trainer):
    def __init__(self, model, tokenizer, unlabeled_df, lambda_u=1.0, mask_prob=0.3, *a, **kw):
        super().__init__(*a, **kw)
        self.model = model
        self.tokenizer = tokenizer
        self.unlabeled_df = unlabeled_df.reset_index(drop=True)
        self.lambda_u = lambda_u
        self.mask_prob = mask_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_model = RobertaForMaskedLM.from_pretrained(MODEL_NAME).to(self.device)
        self.mask_model.eval()

    def bert_mask_augment(self, text):
        words = re.findall(r'\w+|[^\w\s]', text)
        masked = [self.tokenizer.mask_token if np.random.rand()<self.mask_prob and w.isalpha() else w for w in words]
        masked_text = " ".join(masked)
        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        with torch.no_grad(): logits = self.mask_model(**inputs).logits
        for pos in (inputs["input_ids"]==self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]:
            token_id = logits[0, pos].argmax().item()
            words[pos] = self.tokenizer.decode([token_id]).strip()
        return " ".join(words)

    def get_model_outputs(self, model, df, return_loss=False):
        inputs = self.tokenizer(df["sentence"].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        labels = torch.tensor(df["ratings"].astype(float).values).unsqueeze(1).to(self.device)
        labels = torch.cat([1 - labels, labels], dim=1).float()
        outputs = model(**inputs, labels=labels)
        logits = torch.softmax(outputs.logits, dim=-1)
        return (outputs.loss, logits) if return_loss else logits

    def train(self, *a, **kw):
        model = self.model.to(self.device); model.train()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_scheduler("linear", optimizer, 0, NUM_EPOCHS * len(self.train_dataset))
        train_df, unlabeled_df = pd.DataFrame(self.train_dataset), self.unlabeled_df.copy()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for i in range(0, len(train_df), TRAIN_BATCH_SIZE):
                batch = train_df.iloc[i:i+TRAIN_BATCH_SIZE]
                loss_sup, _ = self.get_model_outputs(model, batch, return_loss=True)
                aug_batch = unlabeled_df.sample(min(TRAIN_BATCH_SIZE, len(unlabeled_df)))
                aug_batch["AugText"] = aug_batch["sentence"].apply(self.bert_mask_augment)
                with torch.no_grad(): pseudo = self.get_model_outputs(model, aug_batch)
                aug_batch["ratings"] = (pseudo[:,1] >= 0.5).cpu().numpy().astype(bool)
                aug_loss, _ = self.get_model_outputs(model, aug_batch.rename(columns={"AugText":"sentence"}), return_loss=True)
                total_loss_val = loss_sup + self.lambda_u * aug_loss
                total_loss_val.backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_loss += total_loss_val.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: avg_loss={total_loss/len(train_df):.4f}")
```

---

### 🚀 Training Entry Point

```bash
python run_uda_training_gpu_standalone.py --topics AI --epochs 10
```

Each topic trains separately and logs metrics.

---

## 📊 Metrics Function

```python
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = torch.softmax(torch.tensor(preds), dim=-1).numpy()
    y_pred = (probs[:,1] >= 0.5).astype(int)
    y_true = p.label_ids[:,1] if len(p.label_ids.shape)>1 else p.label_ids
    return {
        "eval_f1": f1_score(y_true, y_pred, average="macro"),
        "eval_precision": precision_score(y_true, y_pred, average="macro"),
        "eval_recall": recall_score(y_true, y_pred, average="macro"),
        "eval_accuracy": accuracy_score(y_true, y_pred),
    }
```

---

## 🧪 Expected Results

| Metric | Range (Small Data) | Notes |
|--------|-------------------|-------|
| Train Loss | 0.6 → 0.3 | Decreasing per epoch |
| F1 | 0.6 – 0.85 | Depending on topic |
| Accuracy | 0.7 – 0.9 | With balanced data |
| F1 = 1.0 | ⚠️ Data leakage | Verify stratified splits |

---

## 🧰 Common Issues

| Error | Meaning | Fix |
|-------|---------|-----|
| `vars() argument must have __dict__` | Dataset passed as dict | Use HuggingFace `Dataset` or pandas DataFrame |
| `batch_size mismatch` | Label tensor size wrong | Ensure `labels` length = input length |
| `No space left on device` | Too many checkpoints | Add `save_total_limit=2` |
| `F1=1.0` | Overlap in splits | Recreate datasets with stratified sampling |

---

## ✅ Recommended Workflow

```bash
python prepare_uda_datasets_from_parquet.py
python run_uda_training_gpu_standalone.py --topics AI --epochs 10
cat uda_results_gpu.csv
```

---

## 🧠 Concept Summary

**UDA Loss Function:**
\[
L_{total} = L_{sup} + \lambda_u L_{unsup}
\]

**Augmentation:** RoBERTa Masked LM fills random `[MASK]` tokens to create perturbations.
Encourages consistency under linguistic noise.

---

## 📎 Folder Structure

```
prepared_uda_datasets/
├── AI/
│   ├── labeled.csv
│   ├── val.csv
│   ├── test.csv
│   └── unlabeled.csv
├── Inflation/
│   ├── labeled.csv
│   ├── val.csv
│   ├── test.csv
│   └── unlabeled.csv
...
```

---

## 💬 Diagnostic Tip

```python
from sklearn.metrics import classification_report
preds = trainer.predict(test_dataset)
print(classification_report(
    preds.label_ids[:,1],
    (preds.predictions.argmax(-1) == 1),
    zero_division=0
))
```

---

### ✅ Summary

This setup:
- Faithfully reproduces your UDA logic
- Uses RoBERTa Masked LM for augmentation
- Is GPU-ready and reproducible
- Compatible with small labeled datasets
