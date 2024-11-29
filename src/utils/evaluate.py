"""
Module for evaluating the model predictions.
- BERTSim: evaluate the model predictions using the BERT for sentence similarity.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils.common import agree
from transformers import BertTokenizer, BertForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(DEVICE)
label2int = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Calcifications": 2, 
    "COPD": 3, 
    "Lung Nodules": 4, 
    "Mesothelioma": 5,
    "Plueral Effusion": 6,
    "Pneumonia": 7, 
    "Pneumothorax": 8, 
    "Tuberculosis": 9
}

def BERTSim(report_A, report_B) -> float:
    """
    Evaluate the model predictions using the BERT for sentence similarity.

    Args:
    - report_A: str, the first report text.
    - report_B: str, the second report text.

    Returns:
    - score: float, the similarity score between the two reports.
    """

    # Tokenize the input reports
    inputs_A = TOKENIZER(report_A, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    inputs_B = TOKENIZER(report_B, return_tensors='pt', padding=True, truncation=True).to(DEVICE)

    # Get the model outputs
    outputs_A = MODEL(**inputs_A, output_hidden_states=True, return_dict=True)
    outputs_B = MODEL(**inputs_B, output_hidden_states=True, return_dict=True)

    # Get the CLS token embeddings
    cls_A = outputs_A["hidden_states"][-1][:, 0, :]
    cls_B = outputs_B["hidden_states"][-1][:, 0, :]

    # Calculate the similarity score
    score = (cls_A @ cls_B.T).item()

    return score


def evaluate_preds(preds, y) -> None:
    """
    Evaluate the model predictions.

    Args:
    - preds
    - y

    Returns:
    - None, prints the evaluation metrics.
    """

    # compute class wise confusion matrices
    conf = np.zeros((10, 2, 2))

    # iteratre over each disease
    for ailment in label2int:
        gt_idx = label2int[ailment]
        for ps, gs in zip(preds, y):
            # True case
            if ailment in gs:
                # TP case
                if ailment in ps:
                    conf[gt_idx, 0, 0] += 1
                # FN case
                else:
                    conf[gt_idx, 0, 1] += 1
            # False case
            else:
                # FP case
                if ailment in ps:
                    conf[gt_idx, 1, 0] += 1
                # TN case
                else:
                    conf[gt_idx, 1, 1] += 1

    # compute class wise accuracy, recall, precision, f1
    acc = np.zeros(10)
    recall = np.zeros(10)
    precision = np.zeros(10)
    f1 = np.zeros(10)

    for i in range(10):
        TP = conf[i, 0, 0]
        FN = conf[i, 0, 1]
        FP = conf[i, 1, 0]
        TN = conf[i, 1, 1]

        acc[i] = (TP + TN) / (TP + TN + FP + FN)
        recall[i] = TP / (TP + FN)
        precision[i] = TP / (TP + FP)
        if TP == 0:
            f1[i] = 0
        else:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    print(f"Class Average Accuracy: {np.mean(acc):.4f}")
    print(f"Class Average Recall: {np.mean(recall):.4f}")
    print(f"Class Average Precision: {np.mean(precision):.4f}")
    print(f"Class Average F1: {np.mean(f1):.4f}")

    return
    

def evaluate_expls(expls, y_es) -> None:
    """
    Evaluate the model explanations.

    Args:
    - expls
    - y_es

    Returns:
    - None, prints the evaluation metrics.
    """

    # compute agree scores
    # compute BERtSim scores
    agree_scores, bert_scores = [], []
    for i in tqdm(range(len(expls)), total=len(expls)):
        agree_score = agree(y_es[i], expls[i])
        bert_score = BERTSim(y_es[i], expls[i])
        agree_scores.append(agree_score)
        bert_scores.append(bert_score)

    print(f"Agree Score: {np.mean(agree_scores):.4f}")
    print(f"Bert Score: {np.mean(bert_scores):.4f}")

    return


if __name__ == "__main__":
    # Get classification scores
    gt = pd.read_csv("data/xray_data.csv")
    disc = pd.read_csv("results/preds/disc_predictions.csv")
    disc.fillna("", inplace=True)
    gpt_pred = pd.read_csv("results/preds/gen_pred_gpt.csv")
    gpt_pred.fillna("", inplace=True)
    med_pred = pd.read_csv("results/preds/gen_pred_med.csv")
    med_pred.fillna("", inplace=True)

    # sort by case
    gt["case"] = gt["case"].apply(lambda x: x.split("_")[1])
    gt.sort_values(by="case", inplace=True)
    gt["case"] = gt["case"].apply(lambda x: f"case_{x}")

    disc["case"] = disc["case"].apply(lambda x: x.split("_")[1])
    disc.sort_values(by="case", inplace=True)
    disc["case"] = disc["case"].apply(lambda x: f"case_{x}")

    gpt_pred["case"] = gpt_pred["case"].apply(lambda x: x.split("_")[1])
    gpt_pred.sort_values(by="case", inplace=True)
    gpt_pred["case"] = gpt_pred["case"].apply(lambda x: f"case_{x}")

    med_pred["case"] = med_pred["case"].apply(lambda x: x.split("_")[1])
    med_pred.sort_values(by="case", inplace=True)
    med_pred["case"] = med_pred["case"].apply(lambda x: f"case_{x}")

    # evaluate predictions
    disc_preds = list(disc["predictions"])
    gpt_preds = list(gpt_pred["prediction"])
    med_preds = list(med_pred["prediction"])
    gt_labels = list(gt["label"])

    evaluate_preds(disc_preds, gt_labels)
    evaluate_preds(gpt_preds, gt_labels)
    evaluate_preds(med_preds, gt_labels)
    
    # Get explanation scores
    gpt_expls = pd.read_csv("results/expls/gen_expl_gpt.csv")
    med_expls = pd.read_csv("results/expls/gen_expl_med.csv")
    gpt1_expls = pd.read_csv("results/expls/gen_expl_gpt_1.csv")
    med_no_ctxt_expls = pd.read_csv("results/expls/gen_expl_med_no_ctxt.csv")
    gpt_no_ctxt_expls = pd.read_csv("results/expls/gen_expl_gpt_no_ctxt.csv")
    disc_gpt = pd.read_csv("results/expls/disc_expl_gpt.csv")
    # disc_med

    gpt_expls.fillna("", inplace=True)
    med_expls.fillna("", inplace=True)
    gpt1_expls.fillna("", inplace=True)
    med_no_ctxt_expls.fillna("", inplace=True)
    gpt_no_ctxt_expls.fillna("", inplace=True)
    disc_gpt.fillna("", inplace=True)

    # sort by case
    gpt_expls["case"] = gpt_expls["case"].apply(lambda x: x.split("_")[1])
    gpt_expls.sort_values(by="case", inplace=True)
    gpt_expls["case"] = gpt_expls["case"].apply(lambda x: f"case_{x}")

    gpt1_expls["case"] = gpt1_expls["case"].apply(lambda x: x.split("_")[1])
    gpt1_expls.sort_values(by="case", inplace=True)
    gpt1_expls["case"] = gpt1_expls["case"].apply(lambda x: f"case_{x}")

    med_expls["case"] = med_expls["case"].apply(lambda x: x.split("_")[1])
    med_expls.sort_values(by="case", inplace=True)
    med_expls["case"] = med_expls["case"].apply(lambda x: f"case_{x}")

    med_no_ctxt_expls["case"] = med_no_ctxt_expls["case"].apply(lambda x: x.split("_")[1])
    med_no_ctxt_expls.sort_values(by="case", inplace=True)
    med_no_ctxt_expls["case"] = med_no_ctxt_expls["case"].apply(lambda x: f"case_{x}")

    gpt_no_ctxt_expls["case"] = gpt_no_ctxt_expls["case"].apply(lambda x: x.split("_")[1])
    gpt_no_ctxt_expls.sort_values(by="case", inplace=True)
    gpt_no_ctxt_expls["case"] = gpt_no_ctxt_expls["case"].apply(lambda x: f"case_{x}")

    disc_gpt["case"] = disc_gpt["case"].apply(lambda x: x.split("_")[1])
    disc_gpt.sort_values(by="case", inplace=True)
    disc_gpt["case"] = disc_gpt["case"].apply(lambda x: f"case_{x}")

    # evaluate explanations
    gpt_expls = list(gpt_expls["explanation"])
    med_expls = list(med_expls["explanation"])
    gpt1_expls = list(gpt1_expls["explanation"])
    med_no_ctxt_expls = list(med_no_ctxt_expls["explanation"])
    gpt_no_ctxt_expls = list(gpt_no_ctxt_expls["explanation"])
    disc_gpt = list(disc_gpt["explanation"])
    gt_expls = list(gt["report"])

    evaluate_expls(gt_expls, gt_expls)
    evaluate_expls(med_no_ctxt_expls, gt_expls)
    evaluate_expls(gpt_no_ctxt_expls, gt_expls)
    evaluate_expls(disc_gpt, gt_expls)
    evaluate_expls(gpt_expls, gt_expls)
    evaluate_expls(gpt1_expls, gt_expls)
    evaluate_expls(med_expls, gt_expls)
