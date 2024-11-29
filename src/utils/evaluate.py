"""
Module for evaluating the model predictions.
- BERTSim: evaluate the model predictions using the BERT for sentence similarity.
"""

import pandas as pd
# from nubia import Nubia
from src.utils.common import match, agree
from transformers import BertTokenizer, BertForSequenceClassification


# NUBIA = Nubia()
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertForSequenceClassification.from_pretrained('bert-base-uncased')


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
    inputs_A = TOKENIZER(report_A, return_tensors='pt', padding=True, truncation=True)
    inputs_B = TOKENIZER(report_B, return_tensors='pt', padding=True, truncation=True)

    # Get the model outputs
    outputs_A = MODEL(**inputs_A, hidden_states=True, return_dict=True)
    outputs_B = MODEL(**inputs_B, hidden_states=True, return_dict=True)

    # Get the CLS token embeddings
    cls_A = outputs_A.hidden_states[-1][:, 0, :]
    cls_B = outputs_B.hidden_states[-1][:, 0, :]

    # Calculate the similarity score
    score = (cls_A @ cls_B.T).item()

    return score


def evaluate_fn(preds, expls, ys, es) -> pd.DataFrame:
    """
    Evaluates the predictions and explanations.

    Args:
        preds (List[str])
            Predictions.
        expls (List[str])
            Explanations.
        ys (List[str])
            Ground truth labels.
        es (List[str])
            Ground truth explanations.

    Returns:
        pd.DataFrame
            Dataframe with the overall evaluation.
    """

    assert len(preds) == len(expls), "[ERROR] Length of predictions and explanations do not match -- aborting evaluation."
    c, i, po, eo = 0, 0, 0, 0
    final_df = {
        "Predictions": [],
        "Explanations": [],
        "Ground Truth": [],
        "Ground Truth Explanation": [],
        "predict_ok": [],
        "explain_ok": [],
        "bert_sim": [],
        # "nub_score": [],
        # "nub_semantic_rel":[],
        # "nub_contradiction":[],
        # "nub_irrelevancy":[],
        # "nub_logical_agreement":[],
        # "nub_grammar_ref":[],
        # "nub_grammar_hyp":[]
    }

    for k, _ in enumerate(preds):
        y_pred, y, e_pred, e = preds[k], ys[k], expls[k], es[k]

        # Check if predictions and explanations are correct
        predict_ok = match(y_pred, y)
        explain_ok = agree(e_pred, e)
        bert_sim = BERTSim(e, e_pred)
        # nubia = NUBIA.score(ref=e,hyp=e_pred, verbose=False, get_features=True)

        # Append to final dataframe
        final_df["Predictions"].append(y_pred)
        final_df["Explanations"].append(e_pred)
        final_df["Ground Truth"].append(y)
        final_df["Ground Truth Explanation"].append(e)
        final_df["predict_ok"].append(predict_ok)
        final_df["explain_ok"].append(explain_ok)
        final_df["bert_sim"].append(bert_sim)
        # final_df["nub_score"].append(nubia["nubia_score"])
        # final_df["nub_semantic_rel"].append(nubia["features"]["semantic_relation"])
        # final_df["nub_contradiction"].append(nubia["features"]["contradiction"])
        # final_df["nub_irrelevancy"].append(nubia["features"]["irrelevancy"])
        # final_df["nub_logical_agreement"].append(nubia["features"]["logical_agreement"])
        # final_df["nub_grammar_ref"].append(nubia["features"]["grammar_ref"])
        # final_df["nub_grammar_hyp"].append(nubia["features"]["grammar_hyp"])

        if predict_ok and explain_ok:
            # Correct
            c += 1
        elif predict_ok and not explain_ok:
            # Prediction correct but explanation incorrect
            po += 1
        elif not predict_ok and explain_ok:
            # Prediction incorrect but explanation correct
            eo += 1
        else:
            # Both incorrect
            i += 1

    print(f"Correct: {c}/{len(preds)}")
    print(f"Prediction correct but explanation incorrect: {po}/{len(preds)}")
    print(f"Prediction incorrect but explanation correct: {eo}/{len(preds)}")
    print(f"Both incorrect: {i}/{len(preds)}")

    return pd.DataFrame(final_df)
