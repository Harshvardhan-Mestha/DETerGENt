"""
Module for evaluating the model predictions.
- BERTSim: evaluate the model predictions using the BERT for sentence similarity.
"""

from transformers import BertTokenizer, BertForSequenceClassification


TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertForSequenceClassification.from_pretrained('bert-base-uncased')


def BERTSim(report_A, report_B) -> float:
    """
    Evaluate the model predictions using the BERT for sentence similarity.

    Args:
    - report_A: str, the first report text.
    - report_B: str, the second report text.

    Returns:
    - score: float, the similarity score between the two reports.
    """

    # Tokenize the input reports
    inputs_A = TOKENIZER(report_A, return_tensors='pt', padding=True, truncation=True)
    inputs_B = TOKENIZER(report_B, return_tensors='pt', padding=True, truncation=True)

    # Get the model outputs
    outputs_A = MODEL(**inputs_A, hidden_states=True, return_dict=True)
    outputs_B = MODEL(**inputs_B, hidden_states=True, return_dict=True)

    # Get the CLS token embeddings
    cls_A = outputs_A.hidden_states[-1][:, 0, :]
    cls_B = outputs_B.hidden_states[-1][:, 0, :]

    # Calculate the similarity score
    score = (cls_A @ cls_B.T).item()

    return score
    