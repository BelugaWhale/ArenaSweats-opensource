import csv
import os

def convert_to_csv(input_filename='raw.txt', output_filename='dataset.csv'):
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            csv_writer = csv.writer(outfile)
            # Write desired header directly (fix typo, remove extras)
            header = ['player_name', 'player_mu', 'player_sigma', 'teammate_name', 
                      'teammate_mu', 'teammate_sigma', 'mu_spread', 'label']
            csv_writer.writerow(header)
            # Skip input header and separator lines
            infile.readline()  # Header
            infile.readline()  # Separator
            for line in infile:
                if not line.strip():
                    continue
                parts = line.split('|')
                cleaned_parts = [p.strip() for p in parts]
                # Extract desired fields: skip game_ts (index 1)
                row_data = cleaned_parts[0:1] + cleaned_parts[2:5] + cleaned_parts[5:8]
                # Label: last field if present (for variable lengths)
                label = cleaned_parts[8] if len(cleaned_parts) > 8 else ''
                row_data.append(label)
                csv_writer.writerow(row_data)
        print(f"Successfully converted '{input_filename}' to '{output_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the conversion function
if __name__ == "__main__":
    convert_to_csv()