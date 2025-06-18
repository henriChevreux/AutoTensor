from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import struct
from google.protobuf.json_format import MessageToDict


def extract_tensorboard_events(event_file_path):
    """Extract comprehensive data from a TensorBoard event file.
    
    This function extracts scalars, tensors, graph definitions, and hyperparameters
    from a TensorBoard event file.
    
    Args:
        event_file_path (str): Path to the TensorBoard event file
        
    Returns:
        dict: Dictionary containing extracted data organized by category
    """
    
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