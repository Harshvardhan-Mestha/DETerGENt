import pandas as pd

csv = pd.read_csv("/home/f20212582/git/radio-lm/data/train_data.csv")
print(len(csv))

valid_labels = ["Atelectasis", "Pneumothorax", "Plueral Effusion", "Cardiomegaly", "Lung Nodules", "Pneumonia"]

csv = csv[csv["label1"].isin(valid_labels)]
print(len(csv))
csv.to_csv("/home/f20212582/git/radio-lm/data/train_data_small.csv", index=False)