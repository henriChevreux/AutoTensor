from openai import OpenAI
from agents import set_default_openai_key
from dotenv import load_dotenv
from pathlib import Path
import os
from tensorboard.backend.event_processing import event_accumulator
from prompts import get_model_analysis_prompt, get_model_code_generation_prompt

def get_gpt_reccomendation_from_tb(context_file_paths):
    all_data = {}
    for version_dir in context_file_paths:
        events_data = extract_tensorboard_events_enhanced(version_dir)
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

def save_model_code_to_file(code_content, filename="generated_model.py"):
    """Save generated model code to a Python file"""
    try:
        with open(filename, 'w') as f:
            f.write(code_content)
        print(f"Model code saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def get_gpt_model_code_from_tb_with_current_model(context_file_paths, current_model_path="model.py"):
    """Get model code from GPT based on TensorBoard analysis and current model"""
    
    # Read the current model code
    try:
        with open(current_model_path, 'r') as f:
            current_model_code = f.read()
    except FileNotFoundError:
        print(f"Warning: Could not find {current_model_path}")
        current_model_code = "Current model code not available"
    
    all_data = {}
    for version_dir in context_file_paths:
        events_data = extract_tensorboard_events_enhanced(version_dir)
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
        temperature=0.1
    )

    return completion.choices[0].message.content

def extract_tensorboard_events_enhanced(event_file_path):
    """Extract comprehensive data from a TensorBoard event file.
    
    This function extracts scalars, tensors, graph definitions, and hyperparameters
    from a TensorBoard event file.
    
    Args:
        event_file_path (str): Path to the TensorBoard event file
        
    Returns:
        dict: Dictionary containing extracted data organized by category
    """
    from tensorboard.backend.event_processing import event_accumulator
    import numpy as np
    import struct
    from google.protobuf.json_format import MessageToDict
    
    # Load the event file with expanded size guidance
    ea = event_accumulator.EventAccumulator(
        event_file_path,
        size_guidance={  # Increase limits to load all data
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
            event_accumulator.GRAPH: 0,
            event_accumulator.AUDIO: 0,
        }
    )
    ea.Reload()  # Load all data
    
    # Get available tags (metrics)
    tags = ea.Tags()
    
    # Dictionary to store all extracted data
    data = {
        'scalars': {},
        'tensors': {},
        'graph': None,
        'hyperparameters': {},
        'metadata': {},
        'text': {},
    }
    
    # Extract scalar metrics
    for tag in tags.get('scalars', []):
        data['scalars'][tag] = ea.Scalars(tag)
    
    # Extract tensor data
    for tag in tags.get('tensors', []):
        tensor_events = ea.Tensors(tag)
        data['tensors'][tag] = []
        
        for tensor_event in tensor_events:
            # Convert tensor proto to more readable format
            tensor_dict = MessageToDict(tensor_event.tensor_proto)
            data['tensors'][tag].append({
                'step': tensor_event.step,
                'wall_time': tensor_event.wall_time,
                'tensor': tensor_dict
            })
    
    # Extract graph definition if available
    if tags.get('graph', []):
        data['graph'] = ea.Graph()
    
    # Extract text summaries (often contain hyperparameters or model descriptions)
    if hasattr(ea, '_plugin_to_tag_to_content'):
        plugin_data = ea._plugin_to_tag_to_content
        
        # Extract text data
        if 'text' in plugin_data:
            for tag, content in plugin_data['text'].items():
                data['text'][tag] = content
        
        # Extract hyperparameters
        if 'hparams' in plugin_data:
            for tag, content in plugin_data['hparams'].items():
                data['hyperparameters'][tag] = content
    
    return data