import csv

# DEFINE THE PATH AND THE CATEGORIZATION TARGET HERE!!!!!
# Don't forget to update OUTCOME_TARGET! (again...)
input_file = 'full_hands.csv'
output_file = 'full_hands_clean.csv'
OUTCOME_TARGET = 4


gestures = {
    0: "pinch",
    1: "point",
    2: "idle",
    3: "twofinger",
    4: "spread"
}


with open(input_file, 'r', newline='') as csvfile_in, open(output_file, 'w', newline='') as csvfile_out:
    reader = csv.reader(csvfile_in)
    writer = csv.writer(csvfile_out)

    for row in reader:
        if len(row) <= 120:
            all_numeric = True
            for value in row:
                try:
                    float(value)
                except ValueError:
                    all_numeric = False
                    break
            
            if all_numeric:
                row.append(OUTCOME_TARGET)
                writer.writerow(row)
