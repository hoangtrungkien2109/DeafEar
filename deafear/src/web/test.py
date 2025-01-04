import os

# Get the base path dynamically
base_path = os.path.dirname(os.path.abspath(__file__))  # This gives the directory of the current script

# Construct the full path to the model
model_path = os.path.join(base_path, 'model', 'model.tflite')

print(model_path)
