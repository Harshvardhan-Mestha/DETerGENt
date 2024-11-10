"""
Interacts with both LLMs. Generates reports and predictions using GPT-4o and MiniGPT-Med.
"""

import os
import sys
from typing import Optional

from openai import OpenAI

from utils import (
    encode_image,
    miniGPTclient,
    PromptArgs,
    MiniGPTArgs,
    MiniGPTv2,
    class_list
)

sys.path.append("minigpt4/")

openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
CLIENT = OpenAI(
    organization=openai_org,
    api_key=openai_key,
)


def generate_report(
    llm: str, x: str,
    model: Optional[MiniGPTv2] = None,
    prompt_args: Optional[PromptArgs] = None,
    model_args: Optional[MiniGPTArgs] = None
) -> str:
    """
    Generate Report for the given X-ray image

    Args
    ----
    llm: str
        Language Model to use (one of 'gpt' or 'medgpt')
    x: str
        Directory of the X-ray image (if multiple images, use the first image) OR the image path

    The next 3 arguments are only used for MiniGPT-Med
    
    model: MiniGPTv2
        MiniGPTv2 model instance
    prompt_args: PromptArgs
        Prompt arguments for the MiniGPT model
    model_args: MiniGPTArgs
        Model arguments (used for running on GPUs)

    Returns
    -------
    report: str
        Generated report for the given X-ray image
    """

    # Get image path
    if os.path.isdir(f"data/{x}"):
        image_path = os.path.abspath(os.listdir(f"data/{x}")[0])
    else:
        image_path = x

    if llm == "gpt":
        print("[INFO] Generating report using GPT-4o")

        # messages object, to be passed to the OpenAI API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given is the X-ray image. 
                        Your task is to output an analysis of the given X-ray with respect to 10 major ailments. 
                        The 10 ailments are as follows : {', '.join(class_list)}
                        You shall analyse this image closely for its attributes. 
                        Your output should be in the following format :
                        "Explanation" - A short passage describing the contents of the image with respect to the ailments above
                        Adhere to the output format strictly, and be concise.
                        Make no assumptions about the patient.
                        """,
                    },
                ],
            }
        ]

        # add the image in base64 encoding
        encoding = encode_image(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoding}"
                },
            }
        )

        valid_report = False
        while not valid_report:
            # get the response from the API
            completion = CLIENT.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
            )
            report = completion.choices[0].message.content
            report = report.replace("\n", " ")
            if not valid_report:
                print("Retrying...")
            elif "Explanation" in report:
                valid_report = True

    elif llm == "medgpt":
        print("[INFO] Generating report using MiniGPT-Med")

        valid_report = False
        while not valid_report:
            # TODO: Refine prompt.
            report_prompt = f"""Describe the contents of the image
             and diagnose the presence of the following diseases: {', '.join(class_list)},
             and write a detailed report on the findings.
            """
            report_mode = "caption"

            # get report from MiniGPTMed
            report, _ = miniGPTclient(
                model_args,
                model,
                image_path,
                report_prompt,
                mode=report_mode,
                temperature=0.9,
                top_p=prompt_args.top_p,
            )

            # TODO: Add conditions to validate the report
            valid_report = True
            if not valid_report:
                print("Retrying...")

    else:
        print("[ERROR] Invalid LLM")
        report = None

    print(report)
    return report


def generate_prediction(
    llm: str, x: str,
    model: Optional[MiniGPTv2] = None,
    prompt_args: Optional[PromptArgs] = None,
    model_args: Optional[MiniGPTArgs] = None
) -> str:

    """
    Get the prediction for the given X-ray image
    Args
    ----
    llm: str
        Language Model to use ('gpt' or 'medgpt')
    x: str
        Directory of the X-ray image (if multiple images, use the first image) OR the image path

    The next 3 arguments are only used for MiniGPT-Med

    model: MiniGPTv2
        MiniGPTv2 model instance
    prompt_args: PromptArgs
        Prompt arguments for the MiniGPT model
    model_args: MiniGPTArgs
        Model arguments (used for running on GPUs)

    Returns
    -------
    y: str
        Predicted ailment for the given X-ray
    """

    # Get image path
    if os.path.isdir(f"data/{x}"):
        image_path = os.path.abspath(os.listdir(f"data/{x}")[0])
    else:
        image_path = x

    if llm == "gpt":
        print("[INFO] Generating predictions using GPT-4o")

        # TODO: refine prompt. (Prob have to do one-by-one)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given is the X-ray image. 
                        Your task is to output an analysis of the given X-ray with respect to 10 major ailments. 
                        You shall analyse this image closely for its attributes. 
                        The 10 ailments are as follows : {', '.join(class_list)}
                        You must provide an output strictly in the following format : 
                        "Diagnosis" - Choose one and only one of the ailments given above
                        Adhere to the output format strictly.
                        Make no assumptions about the patient.
                        Make minimal assumptions.
                """,
                    },
                ],
            }
        ]

        # add the image
        encoding = encode_image(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoding}"},
            }
        )

        valid_prediction = False
        while not valid_prediction:
            # get response from the API
            completion = CLIENT.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=300,
            )
            predictions = completion.choices[0].message.content
            predictions = predictions.replace("\n", " ")

            # check if the prediction is valid and extract the prediction
            if "Diagnosis" in predictions:
                valid_prediction = True
                y = predictions.split("-")[1].strip()

    elif llm == "medgpt":
        print("[INFO] Generating predictions using MiniGPT-Med")

        class_prompt = f"""Which diseases are the most likely to be present in the given Xray?

        {', '.join(class_list)}"""

        valid_prediction = False
        while not valid_prediction:
            class_mode = "vqa"
            label, _ = miniGPTclient(
                model_args,
                model,
                image_path,
                class_prompt,
                mode=class_mode,
                temperature=0.9,
                top_p=prompt_args.top_p,
            )

            # TODO: Add conditions to validate the prediction
            valid_prediction = True
            y = label[0]

    else:
        print("[ERROR] Invalid LLM")
        y = None

    return y
