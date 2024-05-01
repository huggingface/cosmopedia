import os
import logging
import pickle
from collections import Counter

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, make_scorer
from sklearn.model_selection import  train_test_split

SEED = 0
DATASET_NAME = "HuggingFaceTB/llama3_annotations_last_500k_fineweb_2024_parsed"
TARGET_COL = "binary_target"
# llama3_74k_grades_for_classifier
# llama3_20k_grades_for_classifier

logging.basicConfig(level=logging.INFO)

dataset = load_dataset(DATASET_NAME, split="train", cache_dir="/scratch/cosmo/cache/")
logging.info(f"Dataset {DATASET_NAME}: {dataset}")

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


def embed(texts, cache_dir="/fsx/anton/cosmopedia/edu_score/embeddings_cache"):
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


# Prepare the data
embeddings = embed(dataset["text"])
if TARGET_COL not in dataset.column_names:
    dataset = dataset.map(lambda x: {TARGET_COL: 1} if int(x["score"]) > 2.5 else {TARGET_COL: 0})
    dataset.push_to_hub("HuggingFaceTB/llama3_edu_500k_binary_labels", private=True)

X = np.array(embeddings)
y = np.array(dataset[TARGET_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

logging.info(f"Size train, val, test: {len(y_train)}, {len(y_test)}")
logging.info(
    f"label counters:\ntrain:{Counter(y_train)}\ntest: {Counter(y_test)}"
)

# Train
logging.info("Using Logistic Regression")
pipeline = LogisticRegressionCV(Cs=10, cv=5, random_state=42, #class_weight='balanced',
                                scoring=make_scorer(f1_score, average='binary'))

# logging.info("Using Random Forest")
# pipeline = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42)

# logging.info("Using XGBoost")
# pipeline = xgb.XGBClassifier(n_estimators=50, max_depth=5, scale_pos_weight=sum(y_train==0)/sum(y_train==1),
#                       use_label_encoder=False, eval_metric='logloss', random_state=42)

pipeline.fit(X_train, y_train)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Evaluate the model
precision_train = precision_score(y_train, y_pred_train, average="macro")
recall_train = recall_score(y_train, y_pred_train, average="macro")
f1_train = f1_score(y_train, y_pred_train, average="macro")
acc_train = accuracy_score(y_train, y_pred_train)

precision_test = precision_score(y_test, y_pred_test, average="macro")
recall_test = recall_score(y_test, y_pred_test, average="macro")
f1_test = f1_score(y_test, y_pred_test, average="macro")
acc_test = accuracy_score(y_test, y_pred_test)

logging.info("Training Metrics:")
logging.info(f"Accuracy: {acc_train}, Precision: {precision_train}, Recall: {recall_train}, F1 Score: {f1_train}")
logging.info("Test Metrics:")
logging.info(f"Accuracy: {acc_test}, Precision: {precision_test}, Recall: {recall_test}, F1 Score: {f1_test}")
