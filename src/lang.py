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
    organization="org-FS3BNL7yaD4kX7b68zAMckVr",
    api_key="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",
)

CLASS_LIST = ["Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma",
              "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax",
              "Tuberculosis"]


def generate_report(
    llm: str, x: str, pred: Optional[str], no_ctxt: bool,
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
    no_ctxt: bool
        if True, no context is provided, direct explanation
        if False, context is provided, explanation with respect to the prediction

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

    
    if isinstance(pred, float) and no_ctxt == False:
        print("skipped since no ailment diagnosed...")
        report = ""
        return report
    
    if no_ctxt:
        pred = None # if no context is provided, the prediction is not used

    # Get image path
    if os.path.isdir(f"data/{x}"):
        image_path = os.path.abspath(os.listdir(f"data/{x}")[0])
    else:
        image_path = x

    if isinstance(image_path, float) or not os.path.exists(image_path):
        return None

    if llm == "gpt":
        print("[INFO] Generating report using GPT-4o")

        # messages object, to be passed to the OpenAI API
        if no_ctxt and pred is None: # no context mode
            # we do not ask diagnosis we directly explain
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
                            Pretend you are a radiologist
                            Given is the hypothetical X-ray image.
                            The 10 ailments are as follows : "Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma","Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"
                            Your output should be in the following format :
                            "Explanation" - A short passage describing the contents of the image with respect to the ailment(s) above
                            Adhere to the output format strictly, and be concise.
                            Make no assumptions about the patient.
                            """,
                        },
                    ],
                }
            ]
        else:
            # normal mode -- explain with the given pred
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                            Pretend you are a radiologist
                            Given is the hypothetical X-ray image.
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
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
            )
            report = completion.choices[0].message.content
            report = report.replace("\n", " ")
            # print(report)
            if "explanation" in report.lower():
                valid_report = True
            else:
              print("retrying...")

    elif llm == "medgpt":
        print("[INFO] Generating report using MiniGPT-Med")

        if prompt_args is None:
            prompt_args = PromptArgs(temperature=0.9,
                                     top_p=0.9)

        if no_ctxt and pred is None: # no context mode
            # ask for a detailed report with respect to the 10 ailments instead of a diagnosis first
            report_prompt = """
            Please write a detailed report for the given Xray, with respect to "Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma","Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"
            """
        else: # normal mode
            report_prompt = f"""
            Given the prediction {pred}, please write a detailed report for the given Xray.
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
        report = report[0]

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

    if not os.path.exists(image_path):
        return None, None

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
                            "text": f"""
                            Pretend you are a radiologist
                            Given is the hypothetical X-ray image.
                            Your task is to output an analysis of the given X-ray with respect {cls}.
                            You shall analyse this image closely for its attributes.
                            You must provide an output strictly in the following format :
                            "present" - Yes/No
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
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                )
                predictions = completion.choices[0].message.content
                predictions = predictions.replace("\n", " ")
                # print(predictions)
                # check if the prediction is valid and extract the prediction
                if "present" in predictions.lower():
                    valid_prediction = True
                    if "yes" in predictions.lower():
                      ailment = True
                    else:
                      ailment = False


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

        prompt = "Diagnose the given XRay for the presence of {ailment}. Reply in a yes/no manner."
        prompt_args.mode = "vqa"
        y = []
        for ailment in CLASS_LIST:
            label, _ = mini_gpt_client(
                model_args,
                model,
                prompt_args,
                image_path,
                prompt.format(ailment=ailment)
            )

            if label[0].lower() == "yes" or "yes" in label[0].lower():
                y += [ailment]

        y = ", ".join(y)

    else:
        print("[ERROR] Invalid LLM")
        y = None

    print(y)
    return y, image_path
