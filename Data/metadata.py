import os
import xml.etree.ElementTree as ET
import pandas as pd
import json

# Base path to your documents
base_path = '/home/koustav/shreya/disease/xml_dataset/'
errors = []

def extract_specific_trial_metadata_xml(file_path):
    """Extracts specific metadata from an XML document."""
    metadata = {
        'docno': os.path.basename(file_path),
        'min_age': None,
        'max_age': None,
        'gender': None,
        'condition': None,
        'official_title': None,
        'status': None,
        'eligibility_criteria': {
            'inclusion_criteria': None,
            'exclusion_criteria': None
        },
        'condition_browse': None
    }

    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract the specified information
        metadata['max_age'] = root.findtext('eligibility/maximum_age')
        metadata['min_age'] = root.findtext('eligibility/minimum_age')
        metadata['gender'] = root.findtext('eligibility/gender')
        metadata['status'] = root.findtext('overall_status')
        metadata['official_title'] = root.findtext('official_title')

        # Extract eligibility criteria
        criteria_text = root.findtext('eligibility/criteria/textblock')
        if criteria_text:
            # Split into inclusion and exclusion criteria
            criteria_parts = criteria_text.split('Exclusion Criteria:')
            inclusion_criteria = criteria_parts[0].replace('Inclusion Criteria:', '').strip()
            exclusion_criteria = criteria_parts[1].strip() if len(criteria_parts) > 1 else None

            metadata['eligibility_criteria']['inclusion_criteria'] = inclusion_criteria
            metadata['eligibility_criteria']['exclusion_criteria'] = exclusion_criteria

        # Extract condition browse information
        condition_browse = root.find('condition_browse/mesh_term')
        if condition_browse is not None:
            metadata['condition'] = condition_browse.text  # Store in 'condition' instead of 'condition_browse'

        return metadata

    except Exception as e:
        error_message = {
            'file': file_path,
            'error': str(e)
        }
        errors.append(error_message)
        return None  # Return None to signify an error

metadata_list = []

# Walk through the base path and process XML files
for subdir, _, files in os.walk(base_path):
    print(f"Processing folder: {subdir}")
    for file in files:
        if file.endswith('.xml'):  # Check for XML files
            file_path = os.path.join(subdir, file)
            doc_metadata = extract_specific_trial_metadata_xml(file_path)
            if doc_metadata:  # Only append valid metadata
                metadata_list.append(doc_metadata)

# Convert list to DataFrame
metadata_df = pd.DataFrame(metadata_list)

# Check if the DataFrame has the expected columns
if not metadata_df.empty:
    # Output CSV path
    output_csv_path = '/home/koustav/shreya/disease/clinical_trials_metadata_xml.csv'
    metadata_df.to_csv(output_csv_path, index=False)
    
    # Print the first 100 rows with the metadata included
    print(metadata_df.head(100))
else:
    print("No valid metadata found or DataFrame is empty.")

# Save errors to a JSON file
error_json_path = '/home/koustav/shreya/disease/errors_xml.json'
with open(error_json_path, 'w') as error_file:
    json.dump(errors, error_file, indent=4)

# Save metadata to JSON file
metadata_json_path = '/home/koustav/shreya/disease/clinical_trials_metadata_xml.json'
with open(metadata_json_path, 'w') as json_file:
    json_data = {entry['docno']: entry for entry in metadata_list}
    json.dump(json_data, json_file, indent=4)

print(f"Total XML files processed: {len(metadata_df)}")
print(f"Errors recorded in: {error_json_path}")

