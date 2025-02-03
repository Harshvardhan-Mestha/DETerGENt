"""
Simple module that exposes an API call to Qwen.
"""

from typing import List
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

def qwen_client(messages: List[str], model: Qwen2VLForConditionalGeneration, 
                processor: AutoProcessor) -> str:
    """
    This function generates a response from Qwen.

    Args:
        messages (List[str]): List of messages.
        model: The model to use for generation.
        processor: The processor to use for generation.

    Returns:
        output_text (List[str]): List of generated responses.
    """
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text
