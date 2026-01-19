# Importing
import pickle
import subprocess
from pathlib import Path
import re

# Import dictionary (to get class code)
with open("pat_testscores_byclass.pkl", "rb") as f:
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

    # Create a temporary qmd file for each class
    temp_qmd = f"temp_{safe_class_code}.qmd"

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
        
    # Render without --output flags first then move the file
    subprocess.run([
        "quarto", "render", temp_qmd,
        "--to", "pdf",
    ], check=True)

    # Creating PDF as temp_{safe_class_code}.pdf in current directory
    temp_pdf = Path(f"temp_{safe_class_code}.pdf")
    final_pdf = reports_dir / output_file
    
    if temp_pdf.exists():
        import shutil
        shutil.move(str(temp_pdf), str(final_pdf))
        print(f"  ✓ Moved to: {final_pdf}\n")
    else:
        print(f"  ✗ PDF not found: {temp_pdf}\n")
    print(f"Generated report for {class_code}")

    # Clean up temp qmd
    if Path(temp_qmd).exists():
        Path(temp_qmd).unlink()

    # Remove 'temp_classcode_file' folders
    temp_files_dir = Path(f"temp_{safe_class_code}_files")
    if temp_files_dir.exists():
        shutil.rmtree(temp_files_dir)

    # Remove '.tex' files
    temp_tex = Path(f"temp_{safe_class_code}.tex")
    if temp_tex.exists():
        temp_tex.unlink()

    print(f"Generated report for {class_code}")


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