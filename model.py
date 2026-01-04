import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


CLEAN_FILE = "columns_600.csv"
NOISY_FILE = "columns_5000_noisy.csv"
PKL_SAVE = "model.pkl"
SAVE_MODEL = False

df_clean = pd.read_csv(CLEAN_FILE)
df_noisy = pd.read_csv(NOISY_FILE)

df_train_clean, df_test = train_test_split(
    df_clean,
    test_size=0.3,
    stratify=df_clean["normalized_label"],
    random_state=42
)

df_noisy = df_noisy[~df_noisy["map_id"].isin(df_test["column_id"])]
df_noisy.drop(columns=["map_id"], inplace=True)
df_test.drop(columns=["column_id"], inplace=True)
df_train_clean.drop(columns=["column_id"], inplace=True)


vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 5)
)

df_final_train = pd.concat(
    [df_train_clean, df_noisy],
    ignore_index=True
)

base_model = Pipeline([
    ("vec", vectorizer),
    ("clf", LogisticRegression(max_iter=1000))
])

model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=3
)

model.fit(
    df_final_train["column_name"],
    df_final_train["normalized_label"]
)

X_test = df_test["column_name"].astype(str).tolist()
y_test = df_test["normalized_label"].tolist()

probs = model.predict_proba(X_test)
classes = model.classes_

top1_correct = 0
top3_correct = 0

for i, true_label in enumerate(y_test):
    ranked_indices = np.argsort(probs[i])[::-1]
    ranked_labels = [classes[idx] for idx in ranked_indices]

    if ranked_labels[0] == true_label:
        top1_correct += 1

    if true_label in ranked_labels[:3]:
        top3_correct += 1

top1_acc = 100 * top1_correct / len(y_test)
top3_acc = 100 * top3_correct / len(y_test)

print(f"Top-1 accuracy: {round(top1_acc, 2)} %")
print(f"Top-3 accuracy: {round(top3_acc, 2)} %")


if SAVE_MODEL:
    joblib.dump(model, PKL_SAVE)
