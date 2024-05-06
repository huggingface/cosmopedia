import os
import logging
import pickle
import json
from collections import Counter

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score, make_scorer, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

SEED = 0
DATASET_NAME = "HuggingFaceTB/llama3_edu_500k_binary_labels"
TARGET_COL = "score"
# llama3_74k_grades_for_classifier
# llama3_20k_grades_for_classifier

logging.basicConfig(level=logging.INFO)

dataset = load_dataset(DATASET_NAME, split="train", cache_dir="/scratch/cosmo/cache/")
logging.info(f"Dataset {DATASET_NAME}: {dataset}")

embed_model_name = "Snowflake/snowflake-arctic-embed-m"
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
    cache_file = os.path.join(cache_dir, f"{DATASET_NAME.split('/')[1]}_{embed_model_name.split('/')[1]}.pkl")
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
    #dataset = dataset.map(lambda x: {TARGET_COL: 1} if int(x["score"]) > 2.5 else {TARGET_COL: 0})
    #dataset.push_to_hub("HuggingFaceTB/llama3_edu_500k_binary_labels", private=True)

X = np.array(embeddings)
y = np.array(dataset[TARGET_COL])
y = np.clip(y, 0, 5).round().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

logging.info(f"Size train, val, test: {len(y_train)}, {len(y_test)}")
logging.info(
    f"label counters:\ntrain:{Counter(y_train)}\ntest: {Counter(y_test)}"
)

# Train
# logging.info("Using Logistic Regression")
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# pipeline = LogisticRegressionCV(Cs=10, cv=cv, max_iter=1000, random_state=42, class_weight='balanced',
#                                 scoring=make_scorer(f1_score, average='macro'))

# logging.info("Using Ridge Regression")
# pipeline = Ridge(alpha=0.1)

# logging.info("Using Linear Regression")
# pipeline = LinearRegression(
#     fit_intercept=True, copy_X=True, n_jobs=16
# )

logging.info("Using MLP")
pipeline = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, learning_rate_init=5e-5, random_state=42, early_stopping=True, n_iter_no_change=20, verbose=True)

#logging.info("Using MLPRegressor")
#pipeline = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=1e-5, random_state=42, early_stopping=True, n_iter_no_change=20, verbose=True)

# logging.info("Using KNN")
# pipeline = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=16)

# logging.info("Using Random Forest")
# pipeline = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42)

# logging.info("Using XGBoost")
# pipeline = xgb.XGBClassifier(n_estimators=50, max_depth=5, scale_pos_weight=sum(y_train==0)/sum(y_train==1),
#                       use_label_encoder=False, eval_metric='logloss', random_state=42)

pipeline.fit(X_train, y_train)
#print("Best C:", pipeline.C_)
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
y_pred_train = np.clip(y_pred_train, 0, 5).round().astype(int)
y_pred_test = np.clip(y_pred_test, 0, 5).round().astype(int)

logging.info("Training Metrics:")
logging.info(classification_report(y_train, y_pred_train))
logging.info("Test Metrics:")
logging.info(classification_report(y_test, y_pred_test))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

logging.info("Run stats:")
logging.info(json.dumps({
    #'model': {'name': 'LogisticRegressionCV', 'C': pipeline.C_, 'class_weight': pipeline.class_weight},
    #'model': {'name': 'Ridge', 'alpha': 0.1},
    #'model': {'name': 'LinearRegression'},
    'model': {'name': 'MLPClassifier', 'hidden_layer_sizes': [100, 50], 'learning_rate_init': 3e-4},
    #'model': {'name': 'KNN', 'n_neighbors': 5, 'weights': 'distance'},
    #'model': {'name': 'MLPRegressor', 'hidden_layer_sizes': [100], 'learning_rate_init': 3e-4},
    'embedding_model': embed_model_name,
    'train_report': classification_report(y_train, y_pred_train, output_dict=True),
    'test_report': classification_report(y_test, y_pred_test, output_dict=True),
}, cls=NumpyEncoder))


# logging.info("Confusion matrix:")
logging.info(confusion_matrix(y_train, y_pred_train))
logging.info(confusion_matrix(y_test, y_pred_test))