"""
Interacts with both LLMs. Generates reports and predictions using GPT-4o and MiniGPT-Med.
"""

import os
from openai import OpenAI
from typing import Optional, Tuple
from src.utils.common import encode_image
from minigpt4.models.minigpt_v2 import MiniGPTv2
from src.utils.minigpt import mini_gpt_client, PromptArgs, MiniGPTArgs

openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
CLIENT = OpenAI(
    organization=openai_org,
    api_key=openai_key
)

CLASS_LIST = ["Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma",
              "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax",
              "Tuberculosis"]


def generate_report(
    llm: str, x: str, pred: str,
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
    pred: str
        Predicted ailment for the given X-ray

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
        # TODO: refine prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given is the X-ray image.
                        The patient has been diagnosed with {pred}.
                        Your output should be in the following format :
                        "Explanation" - A short passage describing the contents of the image with respect to the ailment(s) above
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

        if prompt_args is None:
            prompt_args = PromptArgs(temperature=0.9,
                                     top_p=0.9)

        valid_report = False
        while not valid_report:
            # TODO: Refine prompt.
            report_prompt = f"""You are a radiologist.
             The patient has been diagnosed with {pred}.
             Given the X-ray image, write a detailed report on the findings.
            """
            prompt_args.mode = "caption"
            # get report from MiniGPTMed
            report, _ = mini_gpt_client(
                model_args,
                model,
                prompt_args,
                image_path,
                report_prompt
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
) -> Tuple[str, str]:
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
        image_name = os.listdir(f"data/{x}")[0]
        image_path = os.getcwd() + f"/data/{x}/" + image_name
    else:
        image_path = x

    if llm == "gpt":
        print("[INFO] Generating predictions using GPT-4o")

        ovr_y = ""
        for cls in CLASS_LIST:
            # TODO: refine prompt.
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Given is the X-ray image.
                            Your task is to output an analysis of the given X-ray with respect {cls}.
                            You shall analyse this image closely for its attributes.
                            You must provide an output strictly in the following format :
                            "Diagnosis" - Yes/No
                            Adhere to the output format strictly.
                            Make no assumptions about the patient.
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

            ailment = False
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
                print(predictions)
                # check if the prediction is valid and extract the prediction
                if "Diagnosis" in predictions:
                    valid_prediction = True
                    ailment = predictions.split("-")[1].strip() == "Yes"

            # add the ailment to the overall prediction
            if ailment:
                ovr_y += cls + ", "

        # remove the trailing comma
        y = ovr_y[:-2]

    elif llm == "medgpt":
        print("[INFO] Generating predictions using MiniGPT-Med")

        if prompt_args is None:
            prompt_args = PromptArgs(temperature=0.0,
                                     top_p=1.0)

        class_prompt = f"""Which diseases are the most likely to be present in the given Xray?
         {', '.join(CLASS_LIST)}"""

        valid_prediction = False
        while not valid_prediction:
            prompt_args.mode = "vqa"
            label, _ = mini_gpt_client(
                model_args,
                model,
                prompt_args,
                image_path,
                class_prompt,
            )

            # TODO: Add conditions to validate the prediction
            valid_prediction = True
            y = label[0]

    else:
        print("[ERROR] Invalid LLM")
        y = None

    return y, image_path
