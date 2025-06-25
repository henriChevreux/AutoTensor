def get_intro_prompt():
    return """
    =================================================================
    🤖 FashionMNIST Training Agent 🤖
    =================================================================
    Welcome! I can help you configure and train FashionMNIST models.
    Type 'help' or '?' to list commands.
    Type 'train' to train with current settings.
    Type 'exit' to quit.
    """

def get_cmd_prompt():
    return '(agent) '

def get_model_analysis_prompt(tb_data):
    return str(f"""
                I have the following TensorBoard event data from my machine learning model training. Each version represents a different run with potentially different model architectures and hyperparameters:

                {tb_data}

                Please analyze this data and provide insights on how to make the model perform better.
                Please also provide a recommendation for the next version of the model. The name of the model class should be the same as the name in the model file.
                Please also provide a recommendation for the hyperparameters for the next version of the model.
                Please also provide a recommendation for the architecture for the next version of the model.
                Please also provide a recommendation for the loss function for the next version of the model.
                Please also provide a recommendation for the optimizer for the next version of the model.
                Please also provide a recommendation for the learning rate for the next version of the model.
                
            """)

def get_model_code_generation_prompt(tb_data, analysis_data, current_model_code):
    return str(f"""
                Here is my current FashionMNIST model code:

                ```python
                {current_model_code}
                ```

                Here is the TensorBoard log data:

                {tb_data}
                
                Here is the analysis of the TensorBoard log data:

                {analysis_data}

                Based on this analysis, please provide an improved version of my model that addresses any issues found in the training patterns.
                The improved model should:
                1. Be a complete, runnable PyTorch Lightning module
                2. Include all necessary imports
                3. Address any overfitting/underfitting issues identified
                4. Optimize the architecture based on the training patterns
                5. Use better hyperparameters if needed

                The name of the model class should be the same as the name in the model file.
                Return ONLY the Python code, no explanations or markdown formatting.
                """)