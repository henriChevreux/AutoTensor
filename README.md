# FashionMNIST Training Agent

An interactive AI agent for configuring, training, and optimizing FashionMNIST models with PyTorch Lightning.

## Features

- Interactive command-line interface for hyperparameter tuning
- Track and compare multiple experiments
- Automatic hyperparameter recommendations based on previous runs
- TensorBoard integration for visualization of training metrics and model activations
- Experiment management and reproducibility
- **Model architecture analysis and generation based on training patterns**

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch and PyTorch Lightning
- TensorBoard
- NumPy
- TensorBoard Python package (required for advanced analytics): `pip install tensorboard`

### Installation

Clone this repository or copy the files to your local machine.

### Running the Agent

To start the hyperparameter agent, run:

```bash
python run_agent.py
```

## Using the Agent

The agent provides a command-line interface with the following commands:

### Basic Commands

- `show` - Display current hyperparameter settings
- `list` - List all available hyperparameters with descriptions
- `info <param>` - Get detailed information about a specific hyperparameter
- `set <param> <value>` - Set a hyperparameter to a specific value
- `reset` - Reset hyperparameters to default values
- `train` - Train a model with the current hyperparameters
- `test [model_path]` - Test the model on the test dataset
- `exit` or `quit` - Exit the program

### Experiment Management

- `compare` - Compare results of all experiments
- `load <experiment_number>` - Load hyperparameters from a previous experiment
- `recommend` - Get recommendations for hyperparameter improvements based on previous experiments
- `load_recommended` - Load the recommended hyperparameter configuration

### Model Architecture Analysis and Generation

- `analyze_model` - Analyze model architecture based on training patterns and suggest improvements
- `generate_model [model_name]` - Generate a new model architecture file based on analysis

## Example Session

```
(agent) show
# Shows current hyperparameter settings

(agent) set learning_rate 0.0005
# Changes the learning rate

(agent) train
# Trains a model with current settings

(agent) analyze_model
# Analyzes model performance and suggests architecture improvements

(agent) generate_model my_improved_model
# Generates a new model architecture file based on analysis

# After implementing a new model, you can train it
(agent) train
# Trains the model with the improved architecture

(agent) compare
# Shows comparison of all trained models
```

## Advanced Model Architecture Features

The agent can analyze TensorBoard logs and model performance to suggest architecture improvements:

1. **Performance Pattern Analysis**: Identifies common issues like overfitting and underfitting through analysis of loss curves and training patterns

2. **Architecture Recommendations**: Suggests architecture modifications based on observed training behavior

3. **Automatic Code Generation**: Creates complete model architecture files tailored to address identified issues:
   - Enhanced architectures for underfitting models
   - Regularized architectures for overfitting models
   - Residual architectures for advanced modeling
   - Data augmentation suggestions

4. **Implementation Guidance**: Provides code examples to implement recommended changes

## Visualizing Results

The agent integrates with TensorBoard for visualization. To view the training metrics and visualizations, run:

```bash
tensorboard --logdir=tb_logs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 