import os
from datetime import datetime
from tensorboard.backend.event_processing import event_accumulator
from gpt_api import get_gpt_analysis_from_tb
import glob

def analysis_pipeline(log_dir="tb_logs", experiment_dir="fashion_mnist"):
    """Main pipeline function for the analysis agent command. 
    Analyze model performance across experiments and identify patterns.
    
    Returns:
        Dictionary with performance analysis and recommendations
    """

    if event_accumulator is None:
        print("TensorBoard not installed. Cannot analyze logs.")
        return None
        
    # Find TensorBoard log directories for the given experiment
    version_dirs = glob.glob(f"{log_dir}/{experiment_dir}/*")
    print(version_dirs)
        
    # Analyze TensorBoard logs using GPT
    tb_analysis = get_tb_logs_analysis(version_dirs)

    # Save analysis to file
    save_analysis_to_file(analysis_dir="analysis_results", version_dirs=version_dirs, results=tb_analysis)

    return tb_analysis

def get_tb_logs_analysis(version_dirs):
    """
    Get the analysis of the TensorBoard logs using GPT.

    Args:
        version_dirs: List of TensorBoard log directories.

    Returns:
        Dictionary of analyzed metrics and patterns
    """

    if not version_dirs:
        print("No TensorBoard logs found.")
        return None
        
    results = get_gpt_analysis_from_tb(version_dirs)
    
    return results

def save_analysis_to_file(analysis_dir="analysis_results", version_dirs=None, results=None):
    """Save the analysis to a file"""
    # Create output directory for analysis results
    os.makedirs(analysis_dir, exist_ok=True)
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"{analysis_dir}/tensorboard_analysis_{timestamp}.txt"
    try:
        # Save results to file with proper formatting
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TENSORBOARD ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TensorBoard Logs Analyzed: {len(version_dirs)} directories\n\n")
            f.write("Log Directories:\n")
            for i, dir_path in enumerate(version_dirs, 1):
                f.write(f"  {i}. {dir_path}\n")
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("GPT ANALYSIS AND RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")
            # Write the GPT results with proper formatting
            if results:
                # Normalize line breaks to ensure proper formatting
                formatted_results = results.replace('\r\n', '\n').replace('\r', '\n')
                f.write(formatted_results)
                # Ensure the file ends with a newline
                if not formatted_results.endswith('\n'):
                    f.write('\n')
            else:
                f.write("No analysis results available.\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF ANALYSIS\n")
            f.write("=" * 80 + "\n")
        print(f"\n✅ TensorBoard analysis saved to: {analysis_file}")
        print(f"📄 Analysis file contains {len(results) if results else 0} characters")
        # Also print the results to console
        print("\n" + "=" * 50)
        print("GPT ANALYSIS RESULTS")
        print("=" * 50)
        print(results)
    except Exception as e:
        print(f"❌ Error saving analysis to file: {e}")
        # Still print results to console even if file saving fails
        print("\n" + "=" * 50)
        print("GPT ANALYSIS RESULTS")
        print("=" * 50)
        print(results)