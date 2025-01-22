import json

# Load the JSON file
input_file = "RCVE_Fixes.json"

# Initialize counters
label_1_count = 0
label_0_count = 0

# Read and process the file line by line
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line)
        if data.get("label") == 1:
            label_1_count += 1
        elif data.get("label") == 0:
            label_0_count += 1

# Print the results
print(f'Number of records with label 1: {label_1_count}')
print(f'Number of records with label 0: {label_0_count}')

output_file = "test_data.json"

# List to store filtered data
filtered_data = []

# Read and filter the data
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line)
            if data.get("label") == 0:
                filtered_data.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

# Write the cleaned data to a new file in a structured JSON format
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered data written to {output_file}")