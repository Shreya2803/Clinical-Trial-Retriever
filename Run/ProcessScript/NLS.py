import pandas as pd
import re
import xml.etree.ElementTree as ET
import nltk
import pyterrier as pt
import json

# Load and parse the XML file
tree = ET.parse('/home/koustav/shreya/disease/topics.xml')
root = tree.getroot()

# Define regex patterns for age and gender
age_pattern = r'\b(\d{1,2})\s*[-]?year[-]?old\b|\b(\d{1,2})\s*[-]?month[-]?old\b|\b(\d{1,2})\s*[-]?week[-]?old\b'
gender_patterns = {
    'male': r'\b(male|man|boy)\b',
    'female': r'\b(female|woman|girl)\b',
    'neutral': r'\b(infant|baby)\b',
}

# Initialize an empty list to store extracted data
data = []

# Iterate over each topic in the XML
for topic in root.findall('topic'):
    topic_number = topic.get('number')
    text = topic.text.strip()
    
    # Extract age
    age_match = re.search(age_pattern, text)
    if age_match:
        if age_match.group(1):  # Year old
            age = int(age_match.group(1))
        elif age_match.group(2):  # Month old
            age = f"{age_match.group(2)} month old"
        elif age_match.group(3):  # Week old
            age = f"{age_match.group(3)} week old"
    else:
        age = None
    
    # Extract gender
    gender = None
    if re.search(gender_patterns['female'], text):
        gender = 'female'
    elif re.search(gender_patterns['male'], text):
        gender = 'male'
    elif re.search(gender_patterns['neutral'], text):
        gender = 'neutral'

    # Append the extracted information to the data list
    data.append({
        'id': topic_number,
        'text': text,
        'age': age,
        'gender': gender
    })

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

def split_text_by_fullstop(text):
    text = text.lower()
    segments = [segment.strip() for segment in text.split('.') if segment.strip()]
    return segments

def minimal_summary(segment):
    keywords = ['boy', 'girl', 'male', 'female', 'man', 'woman', 'diagnosed', 'echocardiography', 'physical',
                'ct scan', 'fluid', 'biopsy', 'testing', 'exam', 'examination', 'laboratory', 'the patient shows',
                'evaluation', 'characteristics', 'medical', 'abnormal', 'ultrasound', 'uterus', 'testes', 'X-ray',
                'endoscopy', 'chronic', 'characterized', 'observed', 'auscultation', 'history', 'mri', 'hemophilia',
                'lipoma', 'otoscopy', 'diagnosis', 'diagnosed', 'mslt']
    
    words = set(segment.lower().split())
    if any(keyword.lower() in words for keyword in keywords):
        if not segment.endswith('.'):
            segment += '.'
        return segment
    return ''

def NLS(df, run_keyword):
    if run_keyword == "Run_1":
        nls_summaries = []
        for text in df['text']:
            segments = split_text_by_fullstop(text)
            summaries = [minimal_summary(segment) for segment in segments if minimal_summary(segment)]
            nls_summary = ' '.join(summaries)
            nls_summaries.append(nls_summary) 
        df['nls_summary'] = nls_summaries  
    return df

df = NLS(df, "Run_1")
print(df.head())
df.to_csv('output.csv', index=False)

# Clean the nls_summary text to avoid parsing issues
def clean_query_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text) 
df['nls_summary'] = df['nls_summary'].apply(clean_query_text)

# Terrier retriever
index_path = '/home/koustav/shreya/disease/Index'
retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)


run_name = "Run_1"
text_lines = []
results_list = []
for idx, row in df.iterrows():
    query = row['nls_summary']
    queries_df = pd.DataFrame({
        'qid': [idx],
        'query': [query]
    })

    try:
        initial_results = retriever.transform(queries_df)

        # Limit to top 1000 results
        top_results = initial_results.head(1000)
        print(f"Results for query ID {idx}:")
        print(top_results.drop(columns=['query']).head(10))

        for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
            text_line = f"{idx} Q0 {result_row['docno']} {rank} {result_row['score']} {run_name[:12]}"
            text_lines.append(text_line)

            results_list.append({
                'topic_no': idx,
                'docno': result_row['docno'],
                'rank': rank,
                'score': result_row['score']
            })

    except Exception as e:
        print(f"An error occurred during retrieval for query ID {idx}: {e}")

# Save the text file with all results
with open(f"{run_name}_NLS_RETRIEVAL.txt", 'w') as txt_file:
    txt_file.write("\n".join(text_lines) + "\n")

# Save the results to a JSON file
with open(f"{run_name}_NLS_RETRIEVAL.json", 'w') as json_file:
    json.dump(results_list, json_file)

