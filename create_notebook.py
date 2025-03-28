import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = [
    nbf.v4.new_code_cell('''import sys
import os

# Add the parent directory to Python path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    print(f"Added {module_path} to Python path")'''),
    
    nbf.v4.new_markdown_cell('# Model Building'),
    
    nbf.v4.new_code_cell('''import pandas as pd
from house_prices.train import build_model

# Load training data
training_data_df = pd.read_csv('../data/train.csv')

# Build and evaluate the model
model_performance_dict = build_model(training_data_df)
print("Model Performance:")
print(model_performance_dict)'''),
    
    nbf.v4.new_markdown_cell('# Model Inference'),
    
    nbf.v4.new_code_cell('''import pandas as pd
from house_prices.inference import make_predictions

# Load test data
user_data_df = pd.read_csv('../data/test.csv')

# Make predictions
predictions = make_predictions(user_data_df)

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': user_data_df['Id'],
    'SalePrice': predictions
})

# Display first few rows of submission
print("First few rows of submission:")
display(submission.head())

# Save submission
submission.to_csv('../submissions/submission.csv', index=False)
print("\\nSubmission file saved successfully!")''')
]

# Add cells to notebook
nb.cells = cells

# Write the notebook to a file
with open('notebooks/model-industrialization-final.ipynb', 'w') as f:
    nbf.write(nb, f) 