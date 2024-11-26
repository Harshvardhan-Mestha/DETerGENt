"""
Main module that generates the predictions and explanations.
"""

import os
import sys
from typing import Optional
from argparse import ArgumentParser

import pandas as pd
from src.lang import generate_report, generate_prediction
from src.utils import load_config, load_data, match, agree


sys.path.append('./')
sys.path.append('minigpt4/')


def main(config, D: list) -> Optional[pd.DataFrame]:
    """
    This function loops through the data and generates predictions and explanations.

    Args:
        config (yaml config): Configuration file (for e.g., see configs/test.yaml).
        D (List): List of data.

    Returns:
        Optional[pd.DataFrame]: Dataframe with the overall evaluation (if evaluation is set to True in the config).
    """

    # checks for config
    pred = config["pred"]
    expl = config["expl"]
    assert pred["from_file"] != pred["generate"], "[INFO] Can either load predictions from file or generate them, not both."
    assert expl["from_file"] != expl["generate"], "[INFO] Can either load explanations from file or generate them, not both."
    if pred["from_file"]:
        assert os.path.exists(pred["path"]), "[INFO] Path to predictions does not exist."
    if expl["from_file"]:
        assert os.path.exists(expl["path"]), "[INFO] Path to explanations does not exist."

    # get predictions
    if pred["from_file"]:
        print(f"[INFO] Loading predictions from file {pred['path']}.")
        preds = pd.read_csv(pred["path"])
        assert "img_dir" in preds.columns or "img_path" in preds.columns, "[ERROR] Image path not found in predictions."
    else:
        print("[INFO] Generating predictions on the fly.")
        print("[INFO] This is only recommended for LLMs and not the Vision models.")
        # TODO: This logic needs rework, need to add some of identification of the case to be predicted
        preds = {
            "img_path": [],
            "pred": [],
            "case": []
        }
        for c in D:
            for pt in c:
                pred = generate_prediction(pred["model"], pt[0])
                preds["img_path"].append(pt[0])
                preds["pred"].append(pred)
                

        preds = pd.DataFrame(preds)
        preds.to_csv(pred["path"])

    # get explanations
    if expl["from_file"]:
        print(f"[INFO] Loading explanations from file {expl['path']}.")
        expls = pd.read_csv(expl["path"])
    else:
        print("[INFO] Generating explanations on the fly.")
        expls = {
            "img_path": [],
            "explanation": [],
            "case": []
        }
        for _, row in preds.iterrows():
            # TODO: remove this print statement
            print(row["img_path"])
            expl = generate_report(expl["model"], row["img_path"], row["pred"])
            expls["img_path"].append(row["img_path"])
            expls["explanation"].append(expl)
            expls["case"].append(row["case"])

        expls = pd.DataFrame(expls)
        expls.to_csv(expl["path"])

    # evaluate
    if config["evaluate"]:
        print("[INFO] Evaluating predictions and explanations.")
        out_csv = evaluate(list(preds["pred"]), list(expls["explanation"]), D[1], D[2])
        return out_csv

    print("[INFO] Skipping evaluation.")
    return None


def evaluate(preds, expls, ys, es) -> pd.DataFrame:
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
        "explain_ok": []
    }

    for k, _ in enumerate(preds):
        y_pred, y, e_pred, e = preds[k], ys[k], expls[k], es[k]

        # Check if predictions and explanations are correct
        predict_ok = match(y_pred, y)
        explain_ok = agree(e_pred, e)

        # Append to final dataframe
        final_df["Predictions"].append(y_pred)
        final_df["Explanations"].append(e_pred)
        final_df["Ground Truth"].append(y)
        final_df["Ground Truth Explanation"].append(e)
        final_df["predict_ok"].append(predict_ok)
        final_df["explain_ok"].append(explain_ok)

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", "--path", type=str, default="./configs/test.yaml")
    args = parser.parse_args()

    CONFIG = load_config(args.p)
    DATA = load_data()

    main(CONFIG, DATA)
