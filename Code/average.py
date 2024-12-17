import pandas as pd
import ast

# Function to convert string representation of dictionary to dictionary
def convert_to_dict(cell):
    return ast.literal_eval(cell)

# Read the CSV file
file_path = './results/evaluated_baseline_with_LLM.csv'
df = pd.read_csv(file_path)

# Apply the conversion function to each cell in the relevant columns
for column in df.columns[1:]:
    df[column] = df[column].apply(convert_to_dict)
# Extract 'Overall_Average' values from each dictionary and calculate the average for each column
overall_averages = df.apply(lambda row: row.apply(lambda x: x['Overall_Average'] if isinstance(x, dict) else None)).mean()

# Print the average 'Overall_Average' for each column
print("Average 'Overall_Average' for each column:")
print(overall_averages)
