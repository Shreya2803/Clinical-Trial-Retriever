import pandas as pd
import re
import xml.etree.ElementTree as ET
import nltk
import pyterrier as pt
import json
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from transformers import T5ForConditionalGeneration, T5Tokenizer
nltk.data.path.append('/home/koustav/shreya/disease/nltk_data')

# Check if the script is being run with a valid argument
if len(sys.argv) != 2:
    print("Usage: python Unipd.py <Run_1|Run_2|Run_3|Run_4>")
    sys.exit(1)

# Get the run mode from the command line argument
run_mode = sys.argv[1]

tree = ET.parse('/home/koustav/shreya/disease/topics.xml')
root = tree.getroot()

# Define regex patterns for age and gender
age_pattern = r'\b(\d{1,2})\s*[-]?year[-]?old\b|\b(\d{1,2})\s*[-]?month[-]?old\b|\b(\d{1,2})\s*[-]?week[-]?old\b'
gender_patterns = {
    'male': r'\b(male|man|boy)\b',
    'female': r'\b(female|woman|girl)\b',
    'neutral': r'\b(infant|baby)\b',
}

data = []

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

    
    data.append({
        'id': topic_number,
        'text': text,
        'age': age,
        'gender': gender
    })

# Create a DataFrame from the extracted data
df = pd.DataFrame(data)

print(df)

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

# Functions for NLS, KS, and RM3 Expansion

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
    if run_keyword in ["Run_1", "Run_3"]:
        nls_summaries = []
        for text in df['text']:
            segments = split_text_by_fullstop(text)
            summaries = [minimal_summary(segment) for segment in segments if minimal_summary(segment)]
            nls_summary = ' '.join(summaries)
            nls_summaries.append(nls_summary) 
        df['nls_summary'] = nls_summaries  
    return df

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

def build_unique_query(text):
    unique_keywords = extract_unique_keywords(text)
    query = ' '.join(unique_keywords)
    return query
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def t5_summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def clean_query_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

# Adjust file name for different run modes
if run_mode == "Run_1":
    df = NLS(df, "Run_1")
    df.to_csv('output_run_1.csv', index=False)
    df['nls_summary'] = df['nls_summary'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))

    # Retrieval using BM25
    index_path = '/home/koustav/shreya/disease/Index'
    retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)
    
    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['nls_summary']
        queries_df = pd.DataFrame({
            'qid': [idx],
            'query': [query]
        })

        try:
            initial_results = retriever.transform(queries_df)
            top_results = initial_results.head(1000)

            for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
                docno = result_row['docno']
                docno = docno.replace('.txt', '')  # This line is indented with 4 spaces
                text_line = f"{query_id} Q0 {docno} {rank} {result_row['score']} Run_1"
                text_lines.append(text_line)
                results_list.append({
                    'topic_no': idx,
                    'docno': result_row['docno'],
                    'rank': rank,
                    'score': result_row['score']
                })
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1

    with open(f"Run_1_NLS_RETRIEVAL_unipd.txt", 'w') as txt_file:
        txt_file.write("\n".join(text_lines) + "\n")

    with open(f"Run_1_NLS_RETRIEVAL_unipd.json", 'w') as json_file:
        json.dump(results_list, json_file)

elif run_mode == "Run_2":
    df['KS_summary'] = df['text'].apply(build_unique_query)
    df.to_csv('output_run_2.csv', index=False)

    # Retrieval using BM25
    index_path = '/home/koustav/shreya/disease/Index'
    retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)
    
    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['KS_summary']
        queries_df = pd.DataFrame({
            'qid': [idx],
            'query': [query]
        })

        try:
            initial_results = retriever.transform(queries_df)
            top_results = initial_results.head(1000)

            for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
                docno = result_row['docno']
                docno = docno.replace('.txt', '')  
                text_line = f"{query_id} Q0 {docno} {rank} {result_row['score']} Run_2"
                text_lines.append(text_line)
                results_list.append({
                    'topic_no': idx,
                    'docno': result_row['docno'],
                    'rank': rank,
                    'score': result_row['score']
                })
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1
    with open(f"Run_2_KS_RETRIEVAL_unipd.txt", 'w') as txt_file:
        txt_file.write("\n".join(text_lines) + "\n")

    with open(f"Run_2_KS_RETRIEVAL_unipd.json", 'w') as json_file:
        json.dump(results_list, json_file)

elif run_mode == "Run_3":
    df = NLS(df, "Run_3")
    df.to_csv('output_run_3.csv', index=False)

    # Cleaning query text for retrieval
    df['nls_summary'] = df['nls_summary'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))

    # Setup RM3 with retrieval
    index_path = '/home/koustav/shreya/disease/Index'
    retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)
    stash = pt.rewrite.stash_results(clear=False)
    rm3_pipe = retriever >> stash >> pt.rewrite.RM3(index_path) >> pt.rewrite.reset_results() >> retriever

    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['nls_summary']
        queries_df = pd.DataFrame({
            'qid': [idx],
            'query': [query]
        })

        try:
            expanded_results = rm3_pipe.transform(queries_df)
            top_results = expanded_results.head(1000)

            for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
                docno = result_row['docno']
                docno = docno.replace('.txt', '')  
                text_line = f"{query_id} Q0 {docno} {rank} {result_row['score']} Run_3"
                text_lines.append(text_line)
                results_list.append({
                    'topic_no': idx,
                    'docno': result_row['docno'],
                    'rank': rank,
                    'score': result_row['score']
                })
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1
    with open(f"Run_3_NLS+RM3_RETRIEVAL_unipd.txt", 'w') as txt_file:
        txt_file.write("\n".join(text_lines) + "\n")

    with open(f"Run_3_NLS+RM3_RETRIEVAL_unipd.json", 'w') as json_file:
        json.dump(results_list, json_file)


elif run_mode == "Run_4":
    df['KS_summary'] = df['text'].apply(build_unique_query)
    df.to_csv('output_run_4.csv', index=False)

    # Retrieval using BM25 and RM3 for Run_4
    index_path = '/home/koustav/shreya/disease/Index'
    retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)

    # Setup RM3 with retrieval
    stash = pt.rewrite.stash_results(clear=False)
    rm3_pipe = retriever >> stash >> pt.rewrite.RM3(index_path) >> pt.rewrite.reset_results() >> retriever

    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['KS_summary']
        queries_df = pd.DataFrame({
            'qid': [idx],
            'query': [query]
        })

        try:
            initial_results = retriever.transform(queries_df)
            expanded_results = rm3_pipe.transform(queries_df)

            # Limit to top 1000 results
            top_results = expanded_results.head(1000)
            for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
                docno = result_row['docno']
                docno = docno.replace('.txt', '') 
                text_line = f"{query_id} Q0 {docno} {rank} {result_row['score']} Run_4"
                text_lines.append(text_line)

                results_list.append({
                    'topic_no': idx,
                    'docno': result_row['docno'],
                    'rank': rank,
                    'score': result_row['score']
                })
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1
    with open(f"Run_4_KS+RM3_RETRIEVAL_unipd.txt", 'w') as txt_file:
        txt_file.write("\n".join(text_lines) + "\n")

    with open(f"Run_4_KS+RM3_RETRIEVAL_unipd.json", 'w') as json_file:
        json.dump(results_list, json_file)
elif run_mode == "Run_5":
    # Apply NLS and then T5 summarization
    df = NLS(df, "Run_1")
    df['t5_summary'] = df['nls_summary'].apply(t5_summarize)
    df['t5_summary'] = df['t5_summary'].apply(clean_query_text)

    # Retrieval using BM25 and T5 summarization
    index_path = '/home/koustav/shreya/disease/Index'
    retriever = pt.terrier.Retriever(index_path, wmodel="BM25", k1=1.2, b=0.75)

    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['t5_summary']
        queries_df = pd.DataFrame({'qid': [idx], 'query': [query]})

        try:
            initial_results = retriever.transform(queries_df)
            top_results = initial_results.head(1000)
            qury_id = 1
            for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
                docno = result_row['docno']
                docno = docno.replace('.txt', '')  # This line is indented with 4 spaces
                text_line = f"{query_id} Q0 {docno} {rank} {result_row['score']} Run_5"
                text_lines.append(text_line)
                results_list.append({'topic_no': idx, 'docno': result_row['docno'], 'rank': rank, 'score': result_row['score']})
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1
    with open(f"Run_5_NLS+T5_RETRIEVAL_unipd.txt", 'w') as txt_file:
        txt_file.write("\n".join(text_lines) + "\n")

    with open(f"Run_5_NLS+T5_RETRIEVAL_unipd.json", 'w') as json_file:
        json.dump(results_list, json_file)
else:
    print(f"Invalid run mode: {run_mode}")
    sys.exit(1)
