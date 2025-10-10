import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def metrics_for(csv_paths):
    """Compute accuracy, precision, recall, and F1 (weighted) for one or more CSVs."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]  # make it iterable if single path

    metrics_list = []

    for path in csv_paths:
        df = pd.read_csv(path)
        y_true = df["gold"]
        y_pred = df["pred"]

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics_list.append({
            "Accuracy": acc,
            "Precision (Weighted)": prec,
            "Recall (Weighted)": rec,
            "F1 (Weighted)": f1,
        })

    # Compute average across runs (if more than one)
    avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
    return avg_metrics


# point these at your prediction CSVs
rows = {
    "Zero-shot (GPT-5)": metrics_for([
        r"..\zero-shot\humaid_zs.csv"
        # add more CSVs here if needed
    ]),
    "BERT (5 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\bert\humaid_bert_ft_5_1\pred_bert_int.csv",
        r"..\artifacts\humaid\bert\humaid_bert_ft_5_2\pred_bert_int.csv",
        r"..\artifacts\humaid\bert\humaid_bert_ft_5_3\pred_bert_int.csv",
    ]),
    "BERTweet (5 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_5_1\pred_bert_int.csv",
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_5_2\pred_bert_int.csv",
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_5_3\pred_bert_int.csv",
    ]),
    "VerifyMatch (5 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\out\humaid_10_10_5_1.csv",
        r"..\artifacts\humaid\out\humaid_10_10_5_2.csv",
        r"..\artifacts\humaid\out\humaid_10_10_5_3.csv",
    ]),
    "BERT (10 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\bert\humaid_bert_ft_10_1\pred_bert_int.csv",
        r"..\artifacts\humaid\bert\humaid_bert_ft_10_2\pred_bert_int.csv",
        r"..\artifacts\humaid\bert\humaid_bert_ft_10_3\pred_bert_int.csv",
    ]),
    "BERTweet (10 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_10_1\pred_bert_int.csv",
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_10_2\pred_bert_int.csv",
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_10_3\pred_bert_int.csv",
    ]),
    "VerifyMatch (10 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\out\humaid_10_10_10_1.csv",
        r"..\artifacts\humaid\out\humaid_10_10_10_2.csv",
        r"..\artifacts\humaid\out\humaid_10_10_10_3.csv",
    ]),
    "BERT (25 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\bert\humaid_bert_ft_25_1\pred_bert_int.csv",
    ]),
    "BERTweet (25 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\bertweet\humaid_bertweet_ft_25_1\pred_bert_int.csv",
    ]),
    "VerifyMatch (25 lb/cl)": metrics_for([
        # add more CSVs if applicable
        r"..\artifacts\humaid\out\humaid_10_10_25_1.csv",
    ]),
}

table = pd.DataFrame(rows).T.round(4)
print(table.to_markdown(index=True))
table.to_csv("humaid_results_table1.csv", index=True)