import os
import glob
from gpt_api import get_gpt_model_code_from_tb_with_current_model

def generate_model_pipeline(
        output_filename = 'generated_model_v2.py', 
        log_dir="tb_logs", 
        experiment_dir="fashion_mnist"):
    """Generate a new model based on TensorBoard analysis and save it as a Python file.
    If no filename is provided, defaults to 'generated_model.py'"""

    print(f"\nAnalyzing TensorBoard logs and generating improved model code...")

    # Get TensorBoard log directories
    log_dir = "tb_logs"
    experiment_dir = "fashion_mnist"
    version_dirs = glob.glob(f"{log_dir}/{experiment_dir}/*")

    if not version_dirs:
        print("No TensorBoard logs found. Cannot generate model without training data.")
        return

    # Generate model code from GPT
    try:
        model_code = get_gpt_model_code_from_tb_with_current_model(version_dirs)

        # Save the code to a file
        saved_file = save_model_code_to_file(model_code, output_filename)
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

def save_model_code_to_file(code_content, filename="generated_model.py"):
    """Save generated model code to a Python file"""
    try:
        with open(os.path.join(os.getcwd(), filename), 'w') as f:
            f.write(code_content)
        print(f"Model code saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving file: {e}")
        return None