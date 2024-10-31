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
import requests
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from medgpt.minigpt4.common.config import Config
from medgpt.minigpt4.models.minigpt_v2 import MiniGPTv2

from typing import List, Tuple
from dataclasses import dataclass




openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    project=openai_project,
    api_key=openai_key,
)

import base64

### GENERAL UTILS

def encode_image(image_path):
    """
    Default encoding for images is base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def summarize(report: str, ailment: str) -> str:
    """
    Generates a summary of the report, in context of the ailment.

    Args:
        report (str): A radiology report
        ailment (str): The ailment to summarize the report for

    Returns:
        str: The summary of the report
    """
    # NOTE: for this task, 3.5 is just as good as any advanced model
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system", 
                "content": "You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax."
                },
            {
                "role": "user",
                "content": "Summarize the given radiology report in context of " 
                + ailment + 
                ". Also, you must (requirement) omit information about the patient age, name, as well as any links. You can also skip the 'report by' information, basically anything not related to the ailment."
                + " Only include information that explicitly mentions the ailment or is close to such a mention."
                + " Strictly do not write 'summary' anywhere, i.e., summarize the report as if you are generating it."
                + " The report is as follows: "
                },
            {
                "role": "user",
                "content": report
                }
            ]
        )

    # parse the response
    summary = completion.choices[0].message.content

    return summary

def match(y, pred) -> bool:
        """
        Match the prediction with the example.

        Args:
            y: ground truth
            pred: prediction

        Returns:
            bool: True if the prediction matches the example, False otherwise
        """
        pred = str(pred).lower()
        y = str(y).lower()
        # pdb.set_trace()

        if ("yes" in pred and "no" not in y) or y == pred:
            return True
        else:
            return False
        
def agree(e, e_pred) -> bool:
        """
        Check if the explanation agrees with the prediction.

        Args:
            e: explanation
            e_pred: explanation

        Returns:
            bool: True if the explanation agrees with the prediction, False otherwise
        """
        # NOTE: for this task, we need to use GPT-4, 3.5 is not enough
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            seed=42,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
                    Your task is to check consistency between two given diagnoses/explanations of an XRay.
                    1. Ignore any personal patient information mentioned in either diagnosis/explanation, e.g. age, name, etc.
                    2. Consider consistency in terms of the symptoms only and not the causes, e.g. if a report mentions xyz can be
                    diagnosed from follow-up and another report just mentions xyz, then this is no problem, it's not necessary to mention follow-up.
                    3. VERY IMPORTANT, your answer should be the same if the two reports are swapped, i.e., independent of the order of the two reports.
                    4. Respond only in Yes/No."""
                },
                {
                    "role": "user",
                    "content": f"Given A: {e} is a diagnosis/explanation of an XRay, and B: {e_pred} is another diagnosis/explanation of an XRay, are these two consistent?"
                },
            ]
        )

        # parse the response
        out = completion.choices[0].message.content.lower()
        if "yes" in out:
            return True
        else:
            return False
        
def parse_response(response, C: Optional[List]) -> Tuple:
        """
        Parse the response from the LLM.

        Args:
            response (str): response from the LLM
            C (List): context

        Returns:
            Tuple: prediction and explanation
        """
        response = response.choices[0].message.content
        pred_and_expl = response.split("\n")
        prediction, explanation = "", ""
        for text in pred_and_expl:
            if "Prediction" in text:
                prediction = text
            if "Explanation" in text:
                explanation = text
        assert prediction != "", "Prediction not found in the response"
        assert explanation != "", "Explanation not found in the response" 
        assert "Prediction" in prediction, "Prediction not found in the response, expected 'Prediction: Yes/No', got " + prediction
        assert "Explanation" in explanation, "Explanation not found in the response, expected 'Explanation: <Your explanation here>', got " + explanation

        # add to the context
        response_conv = {
            "role": "assistant",
            "content": response
        }
        if C != None:
            C.append(response_conv)
        else:
            C = [response_conv]

        return prediction.split(":")[1].strip(), explanation.split(":")[1].strip(), C

def parse():

    parser = ArgumentParser()
    
    # Model loading, always present
    parser.add_argument("--cfg-path", default="medgpt/eval_configs/minigptv2_fp16_eval.yaml")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    
    # experiment specific stuff
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--mode", default="caption")
    parser.add_argument("--expt",default=-1,type=int,help="which expt to run",)
    
    parser.add_argument("--split",default=1,type=float,help="proportion of data to keep in train (default: 1)",)
    parser.add_argument("--max_tries",default=0,type=int,help="max tries for the llm to make (default:3)",)
    parser.add_argument("--pts",default=0,type=int,help="Points to evaluate per class (from both train and test) (default:3)",)
    help_lines = ['classes to evaluate (enter class index in a comma seperated manner)','0:ATCS','1:CLFS','2:CRDM','3:COPD','4:LNGN','5:MSTL','6:PLEF','7:PNUM','8:PNTX','9:TUBC']
    parser.add_argument("--classes",default="0,1,2,3",type=str,help='\n'.join(help_lines))
    args = parser.parse_args()
    
    model_args = MiniGPTArgs(
        options=args.options,
        cfg_path=args.cfg_path,
        gpu_id=args.gpu_id
    )
    
    prompt_args = PromptArgs(
        temperature=args.temperature,
        top_p=args.top_p,
        mode=args.mode,
    )
    
    return prompt_args, model_args, args

### MEDGPT UTILS

@dataclass
class MiniGPTArgs:
    options: List
    cfg_path: str = "medgpt/eval_configs/minigptv2_fp16_eval.yaml"
    gpu_id: int = 0

@dataclass
class PromptArgs:
    temperature: float = 0.6
    top_p: float = 0.9
    mode: str = ""

def load_model(args: MiniGPTArgs) -> MiniGPTv2:
    '''
    Utility function to load a model.
    Complicated because of the way they have made their library.
    '''
    
    assert type(args) == MiniGPTArgs, f"Expect args to have these fields. {MiniGPTArgs.__dict__}"

    conf = Config(args)
    
    device = torch.device("cpu")
    if args.gpu_id != None:
        device = torch.device("cuda")
    model = MiniGPTv2.from_config(conf.model_cfg).to(device=device)
    
    return model

@torch.no_grad()
def generate(args: MiniGPTArgs, 
                   model: MiniGPTv2, 
                   image_path: str, 
                   text: str, 
                   mode="caption", 
                   temperature=1.0, 
                   top_p=0.9) -> Tuple[str, float]:
    '''
    Utility to generate reports with a basic prompt structure
    
    [INST] <Img><ImageHere></Img>{mode}{text} [/INST]
    
    mode can also be empty.
    '''
        
    prompt="[INST] <Img><ImageHere></Img>{mode}{text} [/INST]"
    if mode == "": 
        mode = " " # if empty just add a space
    elif mode[0] != "[":
        mode = f" [{mode}] " # otherwise put these brackets as required
    input_text = prompt.format(mode=mode, text=text)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((448, 448))
    ])
    
    image = transform(Image.open(image_path).convert('RGB'))[None, :]
        
    if args.gpu_id != None:
        image = image.to(dtype=torch.float16, device="cuda")
        
    start = time()
    out = model.generate(image,
                         [input_text], 
                         temperature=temperature, 
                         top_p=top_p)
    end = time()
    
    out, logits = out

    return out, (end - start), logits

def printAndLog(text: str, log: io.FileIO) -> None:
    '''
    Utility that will print and write to passed log.
    '''
    
    print(text)
    log.write(text+"\n")
    
    return

### DATA UTILS

def setup_dirs(t):
    os.mkdir(f"./runs/run_{t}")
    os.mkdir(f"./runs/run_{t}/logs")
    os.mkdir(f"./runs/run_{t}/out")
    os.mkdir(f"./runs/run_{t}/util")

def load_data(data_path="./data/xray_data.csv", split=0.25): # loads data from a csv file
    
    data = pd.read_csv(data_path, index_col=False)

    D_ovr = [
        list(data["img_dir"]),
        list(data["label"]),
        list(data["desc"]),
        list(data["desc_pth"]),
        list(data["diagnosis"]),
        list(data["certainty"]),
        list(data["label_short"]),
    ]
    assert len(data) == len(D_ovr[0]), "Change code back to have len(D_ovr[0])"
    # (x, y, e, extra), converted because easy to use.
    class_idx = []
    class_data = list(data["label_short"])

    # iterate over the dataframe and get indices of classes
    for cname in class_names_short:
        l = []
        for i in range(len(class_data)):
            if cname in class_data[i]:
                l.append(i)
        class_idx.append(l)

    D = []
    for i in range(len(data)):
        D.append([D_ovr[0][i], D_ovr[1][i], D_ovr[2][i]]) # making it a list of [x, y, e]
    D_classwise = []

    # make list of a class wise splits
    for idxs in class_idx:
        D_temp = []
        for i in idxs:
            D_temp.append(D[i])
        D_classwise.append(
            list(
                [D_temp, []]
            )
        )

    return D_classwise

