from openai import OpenAI
from agents import set_default_openai_key
from dotenv import load_dotenv
from pathlib import Path
import os
from tensorboard.backend.event_processing import event_accumulator

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
                "content": "Introduce yourself!"
            },
            {
            "role": "user",
            "content": f"""
                I have the following TensorBoard event data from my machine learning model training. Each version represents a different run with potentially different model architectures and hyperparameters:

                {all_data}

                Please analyze this data and provide insights on how to make the model perform better.
                Please also provide a recommendation for the next version of the model.
                Please also provide a recommendation for the hyperparameters for the next version of the model.
                Please also provide a recommendation for the architecture for the next version of the model.
                Please also provide a recommendation for the loss function for the next version of the model.
                Please also provide a recommendation for the optimizer for the next version of the model.
                Please also provide a recommendation for the learning rate for the next version of the model.
                
            """
            }
        ]
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

def get_epoch_from_filename(filename):
    """Extract epoch information from the filename if available."""
    # filename is like "events.out.tfevents.1234567890.hostname.5"

    return filename[-1]

def process_version_event_files(version_dir):
    """Process all event files in a version and organize by epoch."""
    all_data = {}
    
    for filename in os.listdir(version_dir):
        if "events.out.tfevents" in filename:
            full_path = os.path.join(version_dir, filename)
            
            # Extract epoch info from filename (if available)
            epoch = get_epoch_from_filename(filename)
            
            # Extract data from the event file
            file_data = extract_tensorboard_events_enhanced(full_path)
            
            # Store with epoch information
            all_data[filename] = {
                'epoch': epoch,
                'data': file_data
            }
    
    return all_data