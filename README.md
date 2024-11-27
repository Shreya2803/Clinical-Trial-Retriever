# Clinical-Trial-Retriever
A repository for clinical trial retrieval using sparse retrieval , query summarization , query expansion 
This project implements query expansion techniques for disease diagnosis queries using methods such as NLS (Natural Language Summarization), KS (Keyword Summarization), RM3 (Relevance Model 3), and T5 Summarization. These techniques improve search and retrieval accuracy for medical text-based datasets.After the Retrieval step, we have penalized the score of clinical trials for which a patient is not eligible (Exclusion Criteria : Age and Gender) .

## Features

- **NLS (Named Entity Summarization)**: Extracts key medical terms from patient descriptions.
- **KS (Keyword Summarization)**: Generates unique query terms by filtering important keywords.
- **RM3 (Relevance Model 3)**: Enhances retrieval performance through query expansion.
- **T5 Summarization**: Uses the T5 transformer model to summarize lengthy medical text data.

## Installation

To run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shreya2803/Clinical-Trial-Retriever.git
## Important Installation to Run The Unipd.py file 
```bash
conda create --name disease python=3.8
conda activate disease
conda update -n base -c defaults conda
conda install -c conda-forge openjdk=11
conda env config vars set JAVA_HOME=$(dirname $(dirname $(which java)))  [set the JAVA_HOME environment variable to point to the Java installation:]
restart the Conda environment for the changes to take effect
conda deactivate
conda activate disease
conda install -c huggingface transformers
conda install -c anaconda numpy
conda install -c conda-forge python-terrier nltk scikit-learn
import nltk
nltk.download('stopwords')
nltk.download('punkt')
Conda install nltk
Conda install pandas
Conda install torch
Conda install sentencepiece

