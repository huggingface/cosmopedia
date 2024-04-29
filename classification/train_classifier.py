import os
import logging
import pickle
from collections import Counter
from joblib import dump
from joblib import load

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import make_pipeline


SEED = 0
DATASET_NAME = "HuggingFaceTB/llama3_74k_grades_for_classifier"
TARGET_COL = "binary_target"


logging.basicConfig(level=logging.INFO)

dataset = load_dataset(DATASET_NAME, split="train")
logging.info(f"Dataset {DATASET_NAME}: {dataset}")
# llama3_74k_grades_for_classifier
# llama3_20k_grades_for_classifier

embed_model_name = "mixedbread-ai/mxbai-embed-large-v1"
# embed_model_name = "all-MiniLM-L6-v2"
embed_device = "cuda"
embed_batch_size = 64
embed_max_seq_length = 512

if "mixedbread" in embed_model_name:
    embed_model = SentenceTransformer(embed_model_name, truncate_dim=embed_max_seq_length, device=embed_device)
else:
    embed_model = SentenceTransformer(embed_model_name, device=embed_device)
    embed_model.max_seq_length = embed_max_seq_length

def get_extract(x):
    prompt = x["prompt"]
    extract = prompt.split('Here is an extract from a webpage: "')[1].split('".\n\nWrite an extensive and detailed')[0].strip()
    return {"text_extract": extract}

def embed(texts, cache_dir="/fsx/loubna/projects/cosmopedia/prompts/judge/embeddings_cache", custom_name=None):
    if custom_name:
        cache_file = os.path.join(cache_dir, f"{custom_name}.pkl")
    else:
        cache_file = os.path.join(cache_dir, f"{DATASET_NAME.split('/')[1]}.pkl")
    if os.path.exists(cache_file):
        logging.info("Loading existing embeddings")
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    logging.info("Embedding texts...")
    embeddings = embed_model.encode(
        texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
        logging.info("Embedding saved")
    
    return embeddings


embeddings = embed(dataset["text"])

X = np.array(embeddings)
y = np.array(dataset[TARGET_COL])
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

logging.info(f"Size train, val, test: {len(y_train)}, {len(y_val)}, {len(y_test)}")
logging.info(
    f"label counters:\ntrain:{Counter(y_train)}\nval: {Counter(y_val)}\ntest: {Counter(y_test)}"
)

# Train
logging.info("Using Logistic Regression")
pipeline = make_pipeline(
    # StandardScaler(),
    LogisticRegressionCV(Cs=10, cv=5, max_iter=1000, random_state=42, 
                        scoring=make_scorer(f1_score, average='macro'))
)
pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Evaluate the model
precision_train = precision_score(y_train, y_pred_train, average="macro")
recall_train = recall_score(y_train, y_pred_train, average="macro")
f1_train = f1_score(y_train, y_pred_train, average="macro")

precision_test = precision_score(y_test, y_pred_test, average="macro")
recall_test = recall_score(y_test, y_pred_test, average="macro")
f1_test = f1_score(y_test, y_pred_test, average="macro")

logging.info("Training Metrics:")
logging.info(f"Precision: {precision_train}, Recall: {recall_train}, F1 Score: {f1_train}")
logging.info("Test Metrics:")
logging.info(f"Precision: {precision_test}, Recall: {recall_test}, F1 Score: {f1_test}")

dump(pipeline, "/fsx/loubna/projects/cosmopedia/prompts/judge/embeddings_cache/logistic_regression_binary_classifier.joblib")
logging.info("🎊 Model saved 🎊")
