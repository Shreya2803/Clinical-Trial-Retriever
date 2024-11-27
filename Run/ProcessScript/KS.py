import pandas as pd
import re
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import pyterrier as pt
import json

nltk.data.path.append('/home/koustav/shreya/disease/nltk_data')
nltk.data.path.append('/home/koustav/shreya/disease/nltk_data')
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

# Function to extract unique keywords
def extract_unique_keywords(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    pos_tags = pos_tag(filtered_words)
    important_terms = [word for word, tag in pos_tags if tag in ('NN', 'JJ')]

    seen = set()
    unique_terms = []
    for term in important_terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    return unique_terms

# Function to build the summary query
def build_unique_query(text):
    unique_keywords = extract_unique_keywords(text)
    query = ' '.join(unique_keywords)
    return query

# Apply the keyword-based summarization to each row in the DataFrame
df['KS_summary'] = df['text'].apply(build_unique_query)

# Print the updated DataFrame
print(df)


# Terrier retriever
index_path = '/home/koustav/shreya/disease/Index'
retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)


run_name = "Run_2"
text_lines = []
results_list = []
for idx, row in df.iterrows():
    query = row['KS_summary']
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
with open(f"{run_name}_KS_RETRIEVAL.txt", 'w') as txt_file:
    txt_file.write("\n".join(text_lines) + "\n")

# Save the results to a JSON file
with open(f"{run_name}_KS_RETRIEVAL.json", 'w') as json_file:
    json.dump(results_list, json_file)

