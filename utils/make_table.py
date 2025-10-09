import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def metrics_for(csv_path):
    df = pd.read_csv(csv_path)
    y_true = df["gold"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "Accuracy": acc,
        "Precision (Weighted)": prec,
        "Recall (Weighted)": rec,
        "F1 (Weighted)": f1,
    }

# point these at your prediction CSVs
rows = {
    "Zero-shot (GPT-5)":                metrics_for(r"..\zero-shot\pred_zero_shot_10_8.csv"),
    "BERT (100 lb/cl)":                 metrics_for(r"..\supervised\pred_bert.csv"),
    "VerifyMatch (100 lb/cl)":          metrics_for(r"..\verifymatch\crisismmdinf_10_8_2.csv"),
}

table = pd.DataFrame(rows).T.round(4)
print(table.to_markdown(index=True))
table.to_csv("crisismmd_inf_results_table.csv")
