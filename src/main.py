import yaml
from typing import Tuple, List, Union, Dict, Optional
Prompt = Union[Dict[str, str], List[Dict[str, str]]]

import os
from copy import deepcopy
from openai import OpenAI
from abc import abstractmethod
import sys
sys.path.append('.')
sys.path.append('medgpt')

import io
import os
import torch
import random
import base64

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
# from torchvision import transforms as T
# from medgpt.minigpt4.common.config import Config
# from medgpt.minigpt4.models.minigpt_v2 import MiniGPTv2

from typing import List, Tuple
from dataclasses import dataclass
from argparse import ArgumentParser

from utils import load_config, load_data, match, agree
from lang import generate_report, generate_prediction



openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
# client = OpenAI(
#     organization=openai_org,
#     api_key=openai_key,
# )





def main(config,D):

    pred = config["pred"];               expl = config["expl"];
    pred_model = pred["model"];          expl_model = expl["model"];
    pred_path = pred["path"];            expl_path = expl["path"]
    pred_from_file = pred["from_file"];  expl_from_file = expl["from_file"]; 
    generate_predictionerate = pred["generate"];    expl_generate = expl["generate"] 
      

    assert pred_from_file != generate_predictionerate, "[PRED] Only one of from_file or generate can be True"
    assert expl_from_file != expl_generate, "[EXPL] Only one of from_file or generate can be True"
    assert (pred_from_file and len(pred_path) > 0) or generate_predictionerate, "[PRED] from_file is True but path is not provided"
    assert (expl_from_file and len(expl_path) > 0) or generate_predictionerate, "[EXPL] from_file is True but path is not provided"


    # predict 
    if pred_from_file:
        print("[INFO] Loading predictions from file")
        preds = list(pd.read_csv(pred.path))
    else:
        print("[INFO] Generating predictions")
        preds = []

        for c in D:
            for pt in c[0]:
                x, _, _ = pt
                print(x)
                pred = generate_prediction(pred_model,x)
                preds.append(pred)

        preds = pd.DataFrame(preds)
        preds.to_csv(pred_path)       

        

    # explain
    if expl_from_file:
        print("[INFO] Loading explanations from file")
        expls = list(pd.read_csv(expl.path))
    else:
        print("[INFO] Generating explanations")
        expls = []

        for c in D:
            for pt in c[0]:
                x, _, _ = pt
                print(x)
                expl = generate_report(expl_model,x) ## change to take preds as input
                expls.append(expl)

        expls = pd.DataFrame(expls)
        expls.to_csv(expl_path)

    #evaluate -- can move to seperate file but seems clean here -- more clarity of pipeline


    # print("[INFO] Evaluating")
    assert len(preds) == len(expls), "Length of predictions and explanations do not match -- aborting evaluation"
    gt = pd.read_csv(config["ground_truth_file"])
    ys = gt["label"]
    es = gt["explanation"]
    


    for k in range(len(preds)):
        y_pred = preds[i]; y = ys[i];
        e_pred = expls[i]; e = es[i];
    
        predictOK = match(y_pred,y)
        explainOK = agree(e_pred,e)

        c = 0; i = 0; po = 0; eo = 0; 

        if predictOK and explainOK:
            #Correct
            c += 1
        elif predictOK and not explainOK:
            #Prediction correct but explanation incorrect
            po += 1
        elif not predictOK and explainOK:
            #Prediction incorrect but explanation correct
            eo += 1
        else:
            #Both incorrect
            i += 1

    print(f"Correct: {c}, \n Prediction correct but explanation incorrect: {po}, \nPrediction incorrect but explanation correct: {eo}, \nBoth incorrect: {i}")




    
    
    
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", "--path", type=str, default="./configs/test.yaml")
    args = parser.parse_args()
    
    config = load_config(args.p)
    D = load_data()

    main(config,D)