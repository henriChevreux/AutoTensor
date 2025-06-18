from openai import OpenAI
from agents import set_default_openai_key
from dotenv import load_dotenv
from pathlib import Path
import os
from prompts import get_model_analysis_prompt, get_model_code_generation_prompt
from helpers.tb_helpers import extract_tensorboard_events

def get_gpt_analysis_from_tb(context_file_paths):
    """Get analysis from GPT based on TensorBoard logs.
    
    Args:
        context_file_paths: List of TensorBoard log directories.

    Returns:
        String of the analysis
    """
    
    all_data = {}
    for version_dir in context_file_paths:
        events_data = extract_tensorboard_events(version_dir)
        all_data[version_dir] = events_data

    load_dotenv(Path(".env"))
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        set_default_openai_key(api_key)
    else:
        print("Error: you need to set your OpenAI API key!")

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
            "role": "user",
            "content": get_model_analysis_prompt(all_data)
            }
        ]
    )

    return completion.choices[0].message.content

def get_gpt_model_code_from_tb_with_current_model(context_file_paths, current_model_path="model.py"):
    """Get model code from GPT based on TensorBoard analysis and current model.
    
    Args:
        context_file_paths: List of TensorBoard log directories.
        current_model_path: Path to the current model code.

    Returns:
        String of the model code
    """
    
    # Read the current model code
    try:
        with open(current_model_path, 'r') as f:
            current_model_code = f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find {current_model_path}")
        current_model_code = "Current model code not available"
    
    all_data = {}
    for version_dir in context_file_paths:
        events_data = extract_tensorboard_events(version_dir)
        all_data[version_dir] = events_data

    load_dotenv(Path(".env"))
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        set_default_openai_key(api_key)
    else:
        print("Error: you need to set your OpenAI API key!")

    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "You are an expert machine learning engineer. When asked to provide model code, always return ONLY the Python code without any markdown formatting, comments, or explanations. The code should be ready to run immediately."
            },
            {
                "role": "user",
                "content": get_model_code_generation_prompt(all_data, current_model_code)
            }
        ],
        temperature=0.1 # Reduce randomness in the model's output
    )

    return completion.choices[0].message.content
