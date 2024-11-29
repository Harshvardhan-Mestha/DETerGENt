"""
This module contains common utility functions used across the project.
- encode_image: Encodes image to base64, used for OpenAI API.
- summarize: Generates a summary of the report, in context of the ailment.
- match: Match the prediction with the example.
- agree: Check if two explanations agree.
- print_and_log: Utility that will print and write to `log` file.
"""

import io
import os
import base64
from openai import OpenAI

openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization="org-FS3BNL7yaD4kX7b68zAMckVr",
    api_key="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",
)


CLASS_LIST = ["Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma",
              "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax",
              "Tuberculosis"]
SYS = f"""
You are a radiology expert, with detailed knowledge of {', '.join(CLASS_LIST)}.
Your task is to check factual consistency between two given diagnoses/explanations of an XRay.
1. Ignore any personal patient information mentioned in either diagnosis/explanation, e.g. age, name, etc.
2. Consider consistency in terms of the symptoms only and not the causes, e.g. if a report mentions xyz can be
diagnosed from follow-up and another report just mentions xyz, then this is no problem, it's not necessary to mention follow-up.
3. Respond only in Yes/No."""


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
                You have a detailed knowledge of {', '.join(CLASS_LIST)}."""
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
    yes_count = 0
    yeses = {"A": 0, "B": 0, "C": 0, "D": 0}

    # Pathologies
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        seed=42,
        messages=[
            {
                "role": "system",
                "content": SYS
            },
            {
                "role": "user",
                "content": f"""
                Given A: {e_a} is the correct diagnosis/explanation of an XRay, and B: {e_b} is another diagnosis/explanation of an XRay. 
                Do these talk about the same ailments?
                """
            },
        ]
    )
    out = completion.choices[0].message.content.lower()
    yes_count += out == "yes"
    yeses["A"] += out == "yes"

    # Locations
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        seed=42,
        messages=[
            {
                "role": "system",
                "content": SYS
            },
            {
                "role": "user",
                "content": f"""
                Given A: {e_a} is the correct diagnosis/explanation of an XRay, and B: {e_b} is another diagnosis/explanation of an XRay. 
                Are ailments in A and B located on the same side of the lungs?
                """
            },
        ]
    )
    out = completion.choices[0].message.content.lower()
    yes_count += out == "yes"
    yeses["B"] += out == "yes"

    # Number of pathologies
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        seed=42,
        messages=[
            {
                "role": "system",
                "content": SYS
            },
            {
                "role": "user",
                "content": f"""
                Given A: {e_a} is the correct diagnosis/explanation of an XRay, and B: {e_b} is another diagnosis/explanation of an XRay. 
                Do they talk about the same number of ailments?
                """
            },
        ]
    )
    out = completion.choices[0].message.content.lower()
    yes_count += out == "yes"
    yeses["C"] += out == "yes"

    # Desc. match
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        seed=42,
        messages=[
            {
                "role": "system",
                "content": SYS
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
    out = completion.choices[0].message.content.lower()
    yes_count += out == "yes"
    yeses["D"] += out == "yes"

    return (yes_count / 4), yeses


def print_and_log(text: str, log: io.FileIO) -> None:
    """
    Utility that will print and write to `log` file.
    """

    print(text)
    log.write(text+"\n")
