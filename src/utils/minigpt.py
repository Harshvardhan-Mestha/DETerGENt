"""
This module has utilities for MiniGPT.
- MiniGPTArgs: Dataclass for loading the model to a GPU, and from a config file.
- PromptArgs: Dataclass for setting generation kwargs for MiniGPTMed.
- load_model: Function to load the model.
- mini_gpt_client: Function to interact with the model using a basic prompt structure.
"""

import torch
from time import time
from PIL import Image
from typing import Tuple, List
from dataclasses import dataclass
import torchvision.transforms as T


import sys
sys.path.append('./')
from minigpt4.common.config import Config
from minigpt4.models.minigpt_v2 import MiniGPTv2


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


def load_model(model_args: MiniGPTArgs) -> MiniGPTv2:
    """
    Utility function to load a model.
    Complicated because of the way they have made their library.

    Args:
        args: MiniGPTArgs
            The arguments to load the model (config file, gpu_id, etc.)

    Returns:
        model: MiniGPTv2
            The loaded model.
    """

    assert isinstance(model_args, MiniGPTArgs), f"Expect args to have these fields. {MiniGPTArgs.__dict__}"

    # required to load the model
    conf = Config(model_args)

    device = torch.device("cpu")
    if model_args.gpu_id is not None:
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

    TODO: add support for logits?

    Args:
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

    # prepare the image
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((448, 448))
    ])
    image = transform(Image.open(image_path).convert('RGB'))[None, :]

    # transfer image to GPU if available
    if model_args.gpu_id is not None:
        image = image.to(dtype=torch.float16, device="cuda")

    start = time()
    out = model.generate(image,
                         [input_text],
                         temperature=prompt_args.temperature,
                         top_p=prompt_args.top_p)
    end = time()

    return out, (end - start)
