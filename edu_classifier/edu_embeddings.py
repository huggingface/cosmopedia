import logging
from collections import Counter

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 0


logging.basicConfig(level=logging.INFO)

dataset = load_dataset("HuggingFaceTB/llama3_judge_20k_additive_v2", split="train")
dataset = dataset.filter(lambda x: x["score"] <= 5 and x["score"] >= 0).shuffle(
    seed=SEED
)
# dataset = dataset.shuffle(seed=0).select(range(5_000))
print(dataset)
logging.info(f"Dataset: {dataset}")


embed_model_name = "all-MiniLM-L6-v2"
embed_device = "cuda"
embed_batch_size = 64
embed_max_seq_length = 512

embed_model = SentenceTransformer(embed_model_name, device=embed_device)
embed_model.max_seq_length = embed_max_seq_length


def embed(texts):
    return embed_model.encode(
        texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


logging.info("Embedding texts...")
embeddings = embed(dataset["text"])

X = np.array(embeddings)
y = np.array(dataset["score"])


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

logging.info(f"Size train, val, test: {len(y_train)}, {len(y_val)}, {len(y_test)}")
logging.info(
    f"label counters:\ntrain:{Counter(y_train)}\nval: {Counter(y_val)}\ntest: {Counter(y_test)}"
)

# rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=4,
    random_state=42,
)

logging.info("Fitting the classifier...")
rf.fit(X_train, y_train)

# Make predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Evaluate the model
precision_train = precision_score(y_train, y_pred_train, average="macro")
recall_train = recall_score(y_train, y_pred_train, average="macro")
f1_train = f1_score(y_train, y_pred_train, average="macro")

precision_test = precision_score(y_test, y_pred_test, average="macro")
recall_test = recall_score(y_test, y_pred_test, average="macro")
f1_test = f1_score(y_test, y_pred_test, average="macro")

logging.info("Training Metrics:")
logging.info(
    f"Precision: {precision_train}, Recall: {recall_train}, F1 Score: {f1_train}"
)
logging.info("Test Metrics:")
logging.info(f"Precision: {precision_test}, Recall: {recall_test}, F1 Score: {f1_test}")

# logging.info("Cross-validating...")
# cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')
# logging.info(f"Average CV F1 Score: {np.mean(cv_scores)}")
