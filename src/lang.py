## parser arg list - llm,path,
import sys
sys.path.append('medgpt/')
import os
import time
import base64
import argparse
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# from utils import load_model, generate, printAndLog, MiniGPTArgs
from utils import encode_image, generate, PromptArgs, MiniGPTArgs, load_model
from argparse import ArgumentParser

openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
# client = OpenAI(
#     organization=openai_org,
#     api_key=openai_key,
# )
print("[INFO] Loading MiniGPT-Med, this may take some time if this is the first launch")
# model = load_model(MiniGPTArgs)
model = list


def report_gen(llm,x,model,prompt_args=PromptArgs, model_args=MiniGPTArgs):
    report_prompt = "";
    e_pred = ""

    if llm=='gpt':
        print("[INFO] Generating report using GPT-4o")
        image_paths = []
        for i in [os.listdir(f"./data/" + x)[0]]:
            image_path = f"./data/" + x + "/" + i
            image_paths.append(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given is the X-ray image. Your task is to output an analysis of the given X-ray with respect to 10 major ailments. 
                You shall analyse this image closely for its attributes. 
                The 10 ailments are as follows : "Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma", "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"
                "Explanation" - A short passage describing the contents of the image with respect to the ailments above
                Adhere to the output format strictly, and be concise.
                Make no assumptions about the patient.
                Make minimal assumptions.
                
                """,
                    },
                ],
            }
        ]

        for p in image_paths:
            img = encode_image(p)
            # append a new dict to a list of dicts of list of dicts
            messages[0]["content"].append(
                {"type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                    }
                }
            )

        completion = client.chat.completions.create(model="gpt-4o",messages=messages,max_tokens=300,)
        output = completion.choices[0].message.content
        class_names = ["Atelectasis", "Calcifications", "Cardiomegaly", "COPD", "Lung Nodules", "Mesothelioma", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"]
        given_prompt = messages[0]["content"][0]
        
        print(output);
        
        e = output.split("\n")
        for i in e:
            e_pred = e_pred + i

    if llm=="medgpt":
        print("[INFO] Generating report using MiniGPT-Med")
        try: 
            class_list = ["Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma", "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"]
            classes = ", ".join(class_list)
            report_prompt = f"Describe the contents of the image and diagnose the presence of the following diseases: {classes}, and write a detailed report on the findings";
            report_mode = "caption"
            image_path = f"./data/" + x + "/" + os.listdir(f"./data/" + x)[0]

            
            # report_prompt = f"Given {c}, please write a detailed report for the given Xray."
            report, t1 = generate(model_args, model, image_path, report_prompt, mode=report_mode, 
                                                temperature=0.9, top_p=prompt_args.top_p)

            e_pred = report[0]
        except:
            pass
    print(e_pred)
    return e_pred

def pred_gen(llm,x,model=model,prompt_args=PromptArgs, model_args=MiniGPTArgs):
    class_prompt = "";
    y_pred = "";

    if llm=="gpt":
        print("[INFO] Generating predictions using GPT-4o")
        image_paths = []
        for i in [os.listdir(f"./data/" + x)[0]]:
            image_path = f"./data/" + x + "/" + i
            image_paths.append(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Given is the X-ray image. Your task is to output an analysis of the given X-ray with respect to 10 major ailments. 
                You shall analyse this image closely for its attributes. 
                The 10 ailments are as follows : "Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma", "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"
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

        for p in image_paths:
            img = encode_image(p)
            # append a new dict to a list of dicts of list of dicts
            messages[0]["content"].append(
                {"type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                    }
                }
            )

        completion = client.chat.completions.create(model="gpt-4o",messages=messages,max_tokens=300,)
        output = completion.choices[0].message.content
        class_names = ["Atelectasis", "Calcifications", "Cardiomegaly", "COPD", "Lung Nodules", "Mesothelioma", "Pleural Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"]
        given_prompt = messages[0]["content"][0]
        y = output.split("\n")[0]
        print(output); print(y)

        for c in class_names:
            if c in y: y_pred = c; break

    if llm=="medgpt":
        print("[INFO] Generating predictions using MiniGPT-Med")
        try:
            class_list = ["Atelectasis", "Calcifications", "COPD", "Lung Nodules", "Mesothelioma", "Cardiomegaly", "Plueral Effusion", "Pneumonia", "Pneumothorax", "Tuberculosis"]
            classes = ", ".join(class_list)
            class_prompt = f"Which diseases are the most likely to be present in the given Xray? Atelectasis, Calcifications, COPD,Lung Nodules, Mesothelioma, Cardiomegaly, Pleural Effusion, Pneumonia, Pneumothorax,Tuberculosis";
            class_mode = "vqa"
            image_path = f"./data/" + x + "/" + os.listdir(f"./data/" + x)[0]
            label, t1 = generate(model_args, model, image_path, class_prompt, mode=class_mode, 
                                                temperature=0.9, top_p=prompt_args.top_p)
            
            print(label)
            y_pred = label[0]

            print(y_pred)
        except:
            pass
    return y_pred, class_prompt

