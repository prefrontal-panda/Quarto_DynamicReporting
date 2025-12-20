# Importing
import pickle
import subprocess
from pathlib import Path
import re

# Import dictionary (to get class code)
with open("class_testscores.pkl", "rb") as f:
    class_maths = pickle.load(f)
    class_reading = pickle.load(f)

# Set path to save file
reports_dir = Path(r"C:\Users\Debbie.Chong\Virtual_env\Year10_Reports")
reports_dir.mkdir(parents=True, exist_ok=True)

# Set year
year = 2026

# Printing number of classes
print(f"Number of classes: {len(class_maths)}")

# Read the template file
with open("Class_Dynamic_Report_Draft_v1.0.qmd", "r") as f:
    template_content = f.read()

# Loop for all classes (one pdf per class)
for index, class_code in enumerate(class_maths.keys(), start=1): # Start count from 1
    # Change class code characters to Windows-friendly characters (in the filename)
    safe_class_code = class_code.replace('/', '-').replace('\\', '-').replace(':', '-')
    output_file = f"ClassReport_{safe_class_code}.pdf"

    # Print progress note
    print(f"Processing {index}/{len(class_maths)}: {class_code} -> {output_file}") # Printing index number and class being processed

    # Create a temporary qmd file with updated parameters
    temp_qmd = "temp_class_report.qmd"

    # Replace the default parameter values in Python code
    modified_content = re.sub(
        r'class_code = ".*?"',
        f'class_code = "{class_code}"',
        template_content
    )
    modified_content = re.sub(
        r'year = \d+',
        f'year = {year}',
        modified_content
    )
    
    # Replace the title placeholders in YAML
    modified_content = re.sub(
        r'\{\{params\.class_code\}\}',
        class_code,
        modified_content
    )
    modified_content = re.sub(
        r'\{\{params\.year\}\}',
        str(year),
        modified_content
    )
    
    # Write temporary file
    with open(temp_qmd, "w") as f:
        f.write(modified_content)
    
    # Get output path (to save to correct directory)
    output_path = reports_dir / output_file
    
    # Render without parameters
    subprocess.run([
        "quarto", "render", temp_qmd,
        "--to", "pdf",
        "--output", output_path,
    ], check=True)
    
    print(f"Generated report for {class_code}")

# Clean up temp file
import os
if os.path.exists("temp_class_report.qmd"):
    os.remove("temp_class_report.qmd")


# # Read the template file
# with open("Class_Dynamic_Report_Draft_v1.0.qmd", "r") as f:
#     template_content = f.read()

# # Loop for all classes (one pdf per class)
# for class_code in class_maths.keys():
#     output_file = f"ClassReport_{class_code}.pdf"
    
#     # Create a temporary qmd file with updated parameters
#     temp_qmd = "temp_class_report.qmd"
    
#     # Replace the default parameter values in Python code
#     modified_content = re.sub(
#         r'class_code = ".*?"',
#         f'class_code = "{class_code}"',
#         template_content
#     )
#     modified_content = re.sub(
#         r'year = \d+',
#         f'year = {year}',
#         modified_content
#     )
    
#     # ALSO replace the title placeholders in YAML
#     modified_content = re.sub(
#         r'\{\{params\.class_code\}\}',
#         class_code,
#         modified_content
#     )
#     modified_content = re.sub(
#         r'\{\{params\.year\}\}',
#         str(year),
#         modified_content
#     )
    
#     # Write temporary file
#     with open(temp_qmd, "w") as f:
#         f.write(modified_content)
    
#     # Render without parameters
#     subprocess.run([
#         "quarto", "render", temp_qmd,
#         "--to", "pdf",
#         "--output-dir", str(reports_dir),
#         "--output", output_file,
#     ], check=True)
    
#     print(f"Generated report for {class_code}")

# # Clean up temp file
# import os
# if os.path.exists("temp_class_report.qmd"):
#     os.remove("temp_class_report.qmd")