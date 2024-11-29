"""
Main module that generates the predictions and explanations.
"""

import os
import sys
from typing import Optional
from argparse import ArgumentParser

import pandas as pd
from src.utils.data import load_data
from src.utils.config import load_config
from src.lang import generate_report, generate_prediction
from src.utils.minigpt import load_model, MiniGPTArgs, PromptArgs


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

    # prediction config
    pred = config["pred"]
    # explanation config
    expl = config["expl"]

    # validate config
    assert "out_path" in pred.keys(), "[ERROR] Output path for predictions not found."
    assert "out_path" in expl.keys(), "[ERROR] Output path for explanations not found."
    assert pred["from_file"] != pred["generate"], "[INFO] Can either load predictions from file or generate them, not both."
    assert expl["from_file"] != expl["generate"], "[INFO] Can either load explanations from file or generate them, not both."
    if pred["from_file"]:
        assert os.path.exists(pred["path"]), "[INFO] Path to predictions does not exist."
    if expl["from_file"]:
        assert os.path.exists(expl["path"]), "[INFO] Path to explanations does not exist."

    # create MiniGPTv2Args, MiniGPTArgs, and PromptArgs
    if pred["model"] == "medgpt" or expl["model"] == "medgpt":
        model_args = MiniGPTArgs(**config["medgpt"]["model_args"])
        prompt_args = PromptArgs(**config["medgpt"]["prompt_args"])
        model = load_model(model_args)
    else:
        model_args = None
        prompt_args = None
        model = None

    # get predictions
    if pred["from_file"]:
        print(f"[INFO] Loading predictions from file {pred['path']}.")
        preds = pd.read_csv(pred["path"])
        assert "img_dir" in preds.columns or "img_path" in preds.columns, "[ERROR] Image path not found in predictions."
    else:
        print("[INFO] Generating predictions on the fly.")
        print("[INFO] This is only recommended for LLMs and not the Vision models.")
        preds = {
            "img_path": [],
            "prediction": [],
            "label": [],
            "case": []
        }
        for pt in D:
            prediction, img_path = generate_prediction(pred["model"], pt[0], model, prompt_args, model_args)
            preds["img_path"].append(img_path)
            preds["prediction"].append(prediction)
            preds["label"].append(pt[1])
            preds["case"].append(pt[3])

        preds = pd.DataFrame(preds)
        preds.to_csv(pred["out_path"])

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
        for _, pt in preds.iterrows():
            explanation = generate_report(expl["model"], pt["img_path"], pt["prediction"], expl["no_ctxt"], model, prompt_args, model_args)
            expls["img_path"].append(pt["img_path"])
            expls["explanation"].append(explanation)
            expls["case"].append(pt["case"])
            print(pt["case"])

        expls = pd.DataFrame(expls)
        expls.to_csv(expl["out_path"])

    # # evaluate
    # if config["evaluate"]:
    #     print("[INFO] Evaluating predictions and explanations.")
    #     out_csv = evaluate_fn(list(preds["pred"]), list(expls["explanation"]), D[1], D[2])
    #     return out_csv

    print("[INFO] Skipping evaluation.")
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", "--path", type=str, default="./configs/medgpt_base.yaml")
    args = parser.parse_args()

    CONFIG = load_config(args.p)
    DATA = load_data()

    main(CONFIG, DATA)
