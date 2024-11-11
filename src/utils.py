"""
This module has a lot of utlities, <add utilies>
"""

import io
import os
import sys
import base64
from time import time
from dataclasses import dataclass
from argparse import ArgumentParser
from typing import Tuple, List, Optional

import yaml
import torch
import pandas as pd
from PIL import Image
from openai import OpenAI
from torchvision import transforms as T
from minigpt4.common.config import Config
from minigpt4.models.minigpt_v2 import MiniGPTv2

sys.path.append('.')

class_names_short = ["ATCS", "CLFS", "CRDM", "COPD", "LNGN", "MSTL", "PLEF", "PNUM", "PNTX", "TUBC"]
class_list = ["Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma",
              "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax",
              "Tuberculosis"
              ]

openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    api_key=openai_key,
)


### GENERAL UTILS


def encode_image(image_path):
    """
    Encodes image to base64, used for OpenAI API.
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
                "content": f"""You are a radiology expert. 
                You have a detailed knowledge of {', '.join(class_list)}."""
                },
            {
                "role": "user",
                "content": "Summarize the given radiology report in context of "
                + ailment +
                ". Also, you must (requirement) omit information about the patient age, name, as well as any links. "
                + "You can also skip the 'report by' information, basically anything not related to the ailment. "
                + "Only include information that explicitly mentions the ailment or is close to such a mention. "
                + "Strictly do not write 'summary' anywhere, i.e., summarize the report as if you are generating it. "
                + "The report is as follows: "
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

    # convert to lowercase to avoid case sensitivity
    pred = str(pred).lower()
    y = str(y).lower()

    # check if the prediction is correct
    correct = ("yes" in pred and "no" not in y) or y == pred
    return correct


def agree(e_a, e_b) -> bool:
    """
    Check if two explanations agree.

    Args:
        e_a: an explanation
        e_b: another explanation

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
                "content": f"""
                You are a radiology expert, with detailed knowledge of {', '.join(class_list)}.
                Your task is to check consistency between two given diagnoses/explanations of an XRay.
                1. Ignore any personal patient information mentioned in either diagnosis/explanation, e.g. age, name, etc.
                2. Consider consistency in terms of the symptoms only and not the causes, e.g. if a report mentions xyz can be
                diagnosed from follow-up and another report just mentions xyz, then this is no problem, it's not necessary to mention follow-up.
                3. VERY IMPORTANT, your answer should be the same if the two reports are swapped, i.e., independent of the order of the two reports.
                4. Respond only in Yes/No."""
            },
            {
                "role": "user",
                "content": f"""
                Given A: {e_a} is a diagnosis/explanation of an XRay, and B: {e_b} is another diagnosis/explanation of an XRay. 
                Are these two consistent?
                """
            },
        ]
    )

    # parse the response
    out = completion.choices[0].message.content.lower()

    return out == "yes"


def parse():
    """
    General purpose parse function.
    TODO: Add options available.
    """

    parser = ArgumentParser()

    # Model loading, always present
    parser.add_argument("--cfg-path", default="medgpt/eval_configs/minigptv2_fp16_eval.yaml")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="""override some settings in the used config, the key-value pair
            in xxx=yyy format will be merged into config file (deprecate),
            change to --cfg-options instead."""
    )

    # experiment specific stuff
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--mode", default="caption")
    parser.add_argument("--expt", default=-1, type=int, help="which expt to run")

    parser.add_argument("--split",
                        default=1,
                        type=float,
                        help="proportion of data to keep in train (default: 1)")
    parser.add_argument("--max_tries",
                        default=0,
                        type=int,
                        help="max tries for the llm to make (default:3)")
    parser.add_argument("--pts",
                        default=0,
                        type=int,
                        help="Points to evaluate per class (from both train and test) (default:3)")

    help_lines = ['classes to evaluate (enter class index in a comma seperated manner)',
                  '0:ATCS', '1:CLFS', '2:CRDM', '3:COPD', '4:LNGN', '5:MSTL', '6:PLEF',
                  '7:PNUM', '8:PNTX', '9:TUBC']
    parser.add_argument("--classes", default="0,1,2,3", type=str, help='\n'.join(help_lines))
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
    """
    This object is useful for loading the model to a GPU, and from a config file.
    """
    options: List
    cfg_path: str = "configs/minigpt_default.yaml"
    gpu_id: int = 0


@dataclass
class PromptArgs:
    """
    This object is used to set generation kwargs for MiniGPTMed.
    mode: str
        The mode to use for generation. (default: "caption")
        Can be any string, but if it is not empty, it will be put in brackets.
    temperature: float
        The temperature to use for generation. (default: 1.0)
    top_p: float
        The top_p to use for generation. (default: 0.9)
    """
    temperature: float = 0.6
    top_p: float = 0.9
    mode: str = ""


def load_model(args: MiniGPTArgs) -> MiniGPTv2:
    """
    Utility function to load a model.
    Complicated because of the way they have made their library.
    Args
    ----
    args: MiniGPTArgs
        The arguments to load the model (config file, gpu_id, etc.)

    Returns
    -------
    model: MiniGPTv2
        The loaded model.
    """

    assert isinstance(args, MiniGPTArgs), f"Expect args to have these fields. {MiniGPTArgs.__dict__}"

    conf = Config(args)

    device = torch.device("cpu")
    if args.gpu_id is not None:
        device = torch.device("cuda")
    model = MiniGPTv2.from_config(conf.model_cfg).to(device=device)

    return model


@torch.no_grad()
def mini_gpt_client(model_args: MiniGPTArgs,
                    model: MiniGPTv2,
                    prompt_args: PromptArgs,
                    image_path: str,
                    text: str
) -> Tuple[str, float]:
    """
    Utility to interact with the model using a basic prompt structure
    
    [INST] <Img><ImageHere></Img>{mode}{text} [/INST]

    Args
    ----
    args: MiniGPTArgs
        The arguments to load the model (config file, gpu_id, etc.)
    model: MiniGPTv2
        An instantiated model to use for generation.
    prompt_args: PromptArgs
        The arguments to use for generation:
    image_path: str
        The path to the image to use for generation.
    text: str
        The text to use for generation.

    Returns
    -------
    out: str
        The output from the model.
    """

    prompt="[INST] <Img><ImageHere></Img>{mode}{text} [/INST]"

    mode = prompt_args.mode
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

    if model_args.gpu_id is not None:
        image = image.to(dtype=torch.float16, device="cuda")

    start = time()
    # TODO: check model arguments
    out = model.generate(image,
                         [input_text],
                         temperature=prompt_args.temperature,
                         top_p=prompt_args.top_p)
    end = time()
    # out, logits = out

    return out, (end - start)


def print_and_log(text: str, log: io.FileIO) -> None:
    """
    Utility that will print and write to `log` file.
    """

    print(text)
    log.write(text+"\n")


### DATA UTILS


def load_data(data_path="./data/xray_data.csv",
              split=0.25): # loads data from a csv file
    """
    Loads clas-wise data from a csv file.
    TODO: use split.
    TODO: does this work?

    Args:
        data_path (str, optional)
            Path to the data CSV. Defaults to "./data/xray_data.csv".
            The CSV must have "img_dir", "label", "desc", "desc_pth", "diagnosis", "certainty", "label_short" columns.

        split (float, optional)
            Split ratio. Defaults to 0.25.

    Returns:
        D_classwise: List
            List of class-wise splits
    """

    # read csv
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

    # (x, y, e, extra), converted because easy to use.
    class_idx = []
    class_data = list(data["label_short"])

    # iterate over the dataframe and get indices of classes
    for cname in class_names_short:
        l = []
        for i, cname in enumerate(class_data):
            if cname in class_data[i]:
                l.append(i)
        class_idx.append(l)
        print(f"[INFO] {cname} : {len(l)} examples")

    D = []
    for i in range(len(data)):
        D.append([D_ovr[0][i], D_ovr[1][i], D_ovr[2][i]]) # making it a list of [x, y, e]
    D_classwise = []

    # make list of a class wise splits
    for idxs in class_idx:
        D_temp = []
        for i in idxs:
            D_temp.append(D[i])
        D_classwise.append(list([D_temp, []]))

    print("[INFO] Data loaded.")
    return D_classwise


def load_config(path="./configs/test.yaml"):
    """
    Loads a config file.

    Args:
        path (str, optional)
        Path to config file. Defaults to "./configs/test.yaml".

    Returns:
        config: dict
            The loaded config file.
    """

    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
        print("[INFO] Config loaded")

    return config
