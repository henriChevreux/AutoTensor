import os
import glob
from gpt_api import get_gpt_model_code_from_tb_with_current_model
import logging
from helpers.helpers import generate_filename, get_max_version

def generate_model_pipeline(
        output_folder="models",
        log_dir="tb_logs", 
        experiment_name="fashion_mnist",
        current_model_path=None):
    """Generate a new model based on TensorBoard analysis and save it as a Python file.
    """

    print(f"\nAnalyzing TensorBoard logs and generating improved model code...")

    if current_model_path is None:
        current_model_path = get_latest_model_path(output_folder, experiment_name)

    print(f"Latest model path: {current_model_path}")

    # Suppress TensorBoard logging of warnings to avoid cluttering the output
    logging.getLogger('tensorboard').setLevel(logging.ERROR)
    logging.getLogger('tensorboard.backend.event_processing').setLevel(logging.ERROR)

    # Get TensorBoard log directories
    version_dirs = glob.glob(f"{log_dir}/{experiment_name}/*")
    output_folder = os.path.join(output_folder, experiment_name)

    if not version_dirs:
        print("No TensorBoard logs found. Cannot generate model without training data.")
        return

    # Generate model code from GPT
    try:
        model_code = get_gpt_model_code_from_tb_with_current_model(version_dirs, current_model_path)

        # Create output folder for the model
        os.makedirs(output_folder, exist_ok=True)

        # Generate filename for the model
        output_filename = generate_filename(output_folder, filename_prefix="model", extension="py")

        # Save the code to a file in the output folder
        saved_file = save_model_code_to_file(model_code, output_folder, output_filename)
        if saved_file:
            print(f"\n✅ Model code generated and saved to: {saved_file}")
            print("\nYou can now:")
            print(f"1. Review the code in {saved_file}")
            print("2. Import and use the model in your training pipeline")
            print("3. Run 'train' to test the new model")
        else:
            print("❌ Failed to save model code")

    except Exception as e:
        print(f"❌ Error generating model: {e}")

def save_model_code_to_file(code_content, folder_path, filename="generated_model.py"):
    """Save generated model code to a Python file"""
    try:
        with open(os.path.join(folder_path, filename), 'w') as f:
            f.write(code_content)
        print(f"Model code saved to {os.path.join(folder_path, filename)}")
        return os.path.join(folder_path, filename)
    except Exception as e:
        print(f"Error saving file: {e}")
        return None
    
def get_latest_model_path(models_dir, experiment_name):
    """Get the latest model path from the models directory"""
    models_path = f"{models_dir}/{experiment_name}"
    max_version = get_max_version(models_path, "model", "py")
    return f"{models_dir}/{experiment_name}/model_{max_version}.py"