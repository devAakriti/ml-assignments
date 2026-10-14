# Install: pip install torch torchvision transformers scikit-learn pandas pillow requests
# Dataset (Fakeddit, text + image URLs): https://github.com/entitize/Fakeddit
#   Download multimodal_train.tsv (link given in the repository README)
# Text-only fake news data (optional): https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
import os
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load a sample of the multimodal dataset
df = pd.read_csv("multimodal_train.tsv", sep="\t")
df = df.dropna(subset=["clean_title", "image_url"])
df = df.sample(2000, random_state=42).reset_index(drop=True)
os.makedirs("images", exist_ok=True)

def download(row):
    path = f"images/{row.id}.jpg"
    try:
        if not os.path.exists(path):
            r = requests.get(row.image_url, timeout=5)
            with open(path, "wb") as f:
                f.write(r.content)
        Image.open(path).convert("RGB")
        return path
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        return None

df["path"] = [download(r) for r in df.itertuples()]
df = df.dropna(subset=["path"]).reset_index(drop=True)
y = df["2_way_label"].values
print("Samples with valid image:", len(df), "| label counts:", np.bincount(y))

# 2. Text features using DistilBERT
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()

def text_features(texts, bs=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=64, return_tensors="pt").to(device)
            cls = bert(**enc).last_hidden_state[:, 0, :]
            feats.append(cls.cpu().numpy())
    return np.vstack(feats)

# 3. Image features using ResNet18 (last layer removed)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()
prep = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def image_features(paths, bs=32):
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = torch.stack([prep(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + bs]]).to(device)
            feats.append(resnet(batch).cpu().numpy())
    return np.vstack(feats)

X_text = text_features(df["clean_title"].tolist())
X_img = image_features(df["path"].tolist())

# 4. Early fusion: concatenate text and image features
X_fused = np.hstack([X_text, X_img])
print("Text:", X_text.shape, "Image:", X_img.shape, "Fused:", X_fused.shape)

# 5. Train and compare text-only, image-only and multimodal classifiers
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)

def run(X, name):
    sc = StandardScaler().fit(X[idx_train])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(X[idx_train]), y[idx_train])
    pred = clf.predict(sc.transform(X[idx_test]))
    print(f"{name:12s} Accuracy = {accuracy_score(y[idx_test], pred):.4f}  "
          f"F1 = {f1_score(y[idx_test], pred):.4f}")
    return pred

run(X_text, "Text only")
run(X_img, "Image only")
pred = run(X_fused, "Multimodal")

# 6. Detailed report of the multimodal model (see README for label meaning)
print(classification_report(y[idx_test], pred, target_names=["Label 0", "Label 1"]))
