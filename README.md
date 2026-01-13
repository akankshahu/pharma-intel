# Pharma-Intellect: AI-Powered Drug Discovery Research Assistant

An intelligent, AI-driven research assistant designed to accelerate pharmaceutical R&D by providing evidence-backed answers to complex biomedical questions. Built with Retrieval-Augmented Generation (RAG), this project leverages Large Language Models to synthesize information from PubMed and ClinicalTrials.gov, helping researchers identify drug targets, discover repurposing opportunities, and streamline literature reviews.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Project Phases](#project-phases)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Resources](#resources)
- [License](#license)

---

## Project Overview

### Vision

Pharma-Intellect solves a critical bottleneck in pharmaceutical research: the overwhelming volume of scientific literature and clinical trial data. Instead of manually reviewing thousands of documents, researchers can ask complex questions and receive synthesized, citation-backed answers in seconds.

### Why This Matters

- Time Savings: Reduces literature review time from days to minutes
- Risk Reduction: Helps uncover novel therapeutic pathways and drug repurposing opportunities
- Evidence-Based: Every answer is backed by citations to original sources
- Scalable: Can process thousands of research papers and clinical trials

### Use Case Example

Query: "Summarize recent phase 2 and phase 3 clinical trials that investigated the efficacy of risperidone or its derivatives for treating autoimmune disorders, and list any reported adverse events."

Result: A synthesized answer with direct links to relevant PubMed articles and ClinicalTrials.gov entries.

---

## Features

- Data collection from PubMed and ClinicalTrials.gov APIs
- Semantic search using vector embeddings (Sentence-Transformers)
- Fast vector database storage (ChromaDB)
- LLM-powered answer generation (OpenAI GPT-3.5)
- Interactive web interface (Streamlit)
- Citation tracking and source linking
- Query history management
- Example queries for quick start
- Error handling and logging
- Environment-based configuration

---

## Architecture

### RAG Pipeline

```
User Query
    |
    v
[Embedding] -> Convert query to vector using Sentence-Transformers
    |
    v
[Retrieval] -> Search ChromaDB for top-k similar documents
    |
    v
[Augmentation] -> Combine query + retrieved context
    |
    v
[Generation] -> Send to LLM (OpenAI GPT) for synthesis
    |
    v
[Response] -> Return answer + source citations
```

### Four-Phase Implementation

1. Phase 1: Data Ingestion - Collect data from APIs and save to CSV
2. Phase 2: Knowledge Base - Create embeddings and store in vector database
3. Phase 3: RAG Engine - Build query engine with LLM integration
4. Phase 4: Web Interface - Interactive Streamlit application

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.13 | Core programming language |
| Web Framework | Streamlit | Interactive web interface |
| Data Sources | PubMed & ClinicalTrials.gov APIs | Biomedical literature and trial data |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) | Text-to-vector conversion |
| Vector Database | ChromaDB | Fast semantic search storage |
| LLM | OpenAI GPT-3.5-turbo | Answer generation |
| Data Processing | Pandas | ETL and data transformation |
| HTTP Requests | Requests library | API communication |
| Configuration | python-dotenv | Environment variable management |

---

## Project Structure

```
pharma_intellect_project/
|-- README.md                              (This file)
|-- requirements.txt                       (Python dependencies)
|-- .gitignore                             (Git ignore rules)
|-- .env.example                           (Environment template)
|-- config.py                              (Configuration module)
|-- LICENSE                                (MIT License)
|
|-- Phase_1_Data_Collection/
|   |-- data_collector.py                  (Fetch from APIs)
|   |-- fetch_pubmed_alternative.py        (Alternative PubMed source)
|   |-- data/
|   |   |-- pubmed_data.csv
|   |   |-- clinical_trials_data.csv
|   |   |-- .gitkeep
|
|-- Phase_2_Knowledge_Base/
|   |-- embeddings_creator.py              (Generate embeddings)
|   |-- quick_embeddings.py                (Demo version)
|   |-- chroma_db/                         (Vector database)
|
|-- Phase_3_RAG_Engine/
|   |-- rag_engine.py                      (RAG query engine)
|
|-- Phase_4_UI/
    |-- app.py                             (Streamlit interface)
```

---

## Prerequisites

- Python 3.11 or higher (tested with 3.13)
- pip package manager
- Internet connection (for API calls)
- OpenAI API key (for LLM access)
- At least 2GB RAM

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sharmaharshitnone/Pharma-Intellect.git
cd pharma_intellect_project
```

### Step 2: Create Virtual Environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

You should see (env) in your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
pip list | grep -E "requests|pandas|sentence-transformers|streamlit|chromadb|openai"
```

All packages should be listed.

---

## Configuration

### Set Up Your OpenAI API Key

Option 1: Export as environment variable
```bash
export OPENAI_API_KEY="sk-proj-your-actual-key"
```

Option 2: Create .env file from template
```bash
cp .env.example .env
# Edit .env with your API key
nano .env
```

Then add your key:
```
OPENAI_API_KEY=sk-proj-your-actual-key
```

Option 3: Load from Python code
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

---

## How to Run

### Quick Start (Full Pipeline)

```bash
# 1. Activate environment
source env/bin/activate

# 2. Collect data
python Phase_1_Data_Collection/data_collector.py

# 3. Create embeddings
python Phase_2_Knowledge_Base/quick_embeddings.py

# 4. Launch web interface
streamlit run Phase_4_UI/app.py
```

Then open http://localhost:8501 in your browser.

### Option 1: Run Individual Phases

#### Phase 1 - Data Collection
```bash
python Phase_1_Data_Collection/data_collector.py
```

Expected output: CSV files with PubMed articles and clinical trials

#### Phase 2 - Embeddings
```bash
python Phase_2_Knowledge_Base/quick_embeddings.py
```

Expected output: ChromaDB vector database initialized

#### Phase 3 - Test RAG Engine
```bash
python Phase_3_RAG_Engine/rag_engine.py
```

Expected output: Sample query result with answer and sources

#### Phase 4 - Web Interface
```bash
streamlit run Phase_4_UI/app.py
```

Opens interactive web interface at http://localhost:8501

### Option 2: Run Web Interface Only

If data and embeddings already exist:
```bash
streamlit run Phase_4_UI/app.py
```

### Web Interface Usage

1. Enter your research question in the text area
2. Adjust the number of sources (3-15 documents)
3. Click "Get Answer"
4. View the synthesized answer
5. Expand sources to see detailed citations
6. Click links to access original sources

Example questions:
- "What are the latest clinical trials for treating autoimmune disorders?"
- "Summarize recent findings on drug repurposing for oncology"
- "What adverse events have been reported for risperidone?"

---

## Project Phases

### Phase 1: Data Ingestion (ETL)

Goal: Collect raw biomedical data from public APIs and transform into CSV files.

Data Sources:
- PubMed API: Biomedical literature abstracts
- ClinicalTrials.gov API v2: Clinical trial information

Key Functions:
- fetch_pubmed_data(): Query PubMed API
- fetch_clinical_trials_data(): Query ClinicalTrials.gov API
- clean_and_deduplicate(): Remove duplicates

Output: CSV files with structured data
- pubmed_data.csv: 100+ articles with title, abstract, PMID, URL
- clinical_trials_data.csv: 474+ trials with title, status, phase, outcomes

To Run:
```bash
python Phase_1_Data_Collection/data_collector.py
```

### Phase 2: Knowledge Base (Vector Embeddings)

Goal: Convert text to vectors and store in ChromaDB for semantic search.

Process:
1. Load CSV files from Phase 1
2. Text chunking (300 chars with 50 char overlap)
3. Generate embeddings using Sentence-Transformers
4. Store in ChromaDB collections

Collections:
- pubmed_abstracts: PubMed articles
- clinical_trials: Clinical trial information

To Run:
```bash
python Phase_2_Knowledge_Base/quick_embeddings.py
```

### Phase 3: RAG Engine

Goal: Create query engine that retrieves documents and generates answers.

Process:
1. Encode user query to vector
2. Search ChromaDB collections
3. Retrieve top-k most relevant documents
4. Send query + context to OpenAI LLM
5. Return synthesized answer with sources

Main Class: PharmaIntellectRAG
- retrieve_documents(query, num_results): Get relevant documents
- generate_answer(query, retrieved_docs): Generate LLM answer
- query(question, num_results): Complete query execution

To Run:
```bash
python Phase_3_RAG_Engine/rag_engine.py
```

### Phase 4: Web Interface

Goal: Provide interactive user interface for queries.

Features:
- Text input for research questions
- Adjustable source count (3-15)
- Pre-loaded example queries
- Answer display with formatting
- Expandable source cards with metadata
- Query history tracking

To Run:
```bash
streamlit run Phase_4_UI/app.py
```

Then open http://localhost:8501

---

## Usage Examples

### Example 1: Drug Repurposing Research

Input: "What clinical trials have investigated risperidone for treating autoimmune disorders?"

Output:
- Synthesized answer from relevant trials
- Links to each clinical trial
- Publication dates and trial status
- Reported efficacy and adverse events

### Example 2: Literature Review

Input: "Summarize the latest findings on oncology drug discovery targeting immune checkpoints."

Output:
- Summary of recent research
- Links to PubMed articles
- Publication dates
- Key findings and mechanisms

### Example 3: Adverse Events Search

Input: "What adverse events have been reported in phase 3 trials for Valsartan in hypertension?"

Output:
- List of reported adverse events
- Frequency and severity
- Clinical trial references
- Direct links to trial data

---

## Troubleshooting

### Issue: OpenAI API key not found

Solution:
```bash
# Check if OPENAI_API_KEY is set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="sk-proj-your-key"
```

### Issue: ChromaDB collection not found

Solution:
```bash
# Regenerate embeddings
python Phase_2_Knowledge_Base/quick_embeddings.py

# Or check if chroma_db directory exists
ls -la chroma_db/
```

### Issue: PubMed data not loading

Solution:
```bash
# Test API connectivity
curl -X GET "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=oncology&retmax=5&rettype=json"

# Check internet connection
ping 8.8.8.8
```

### Issue: Slow embeddings generation

Solution:
- Use quick_embeddings.py instead of full embeddings_creator.py
- Use GPU acceleration if available
- Reduce number of documents processed

### Issue: Port 8501 already in use

Solution:
```bash
# Use different port
streamlit run Phase_4_UI/app.py --server.port 8502
```

### Issue: Memory errors

Solution:
- Close other applications
- Use quick_embeddings.py for demo
- Reduce NUM_PUBMED and NUM_TRIALS in quick_embeddings.py
- Increase system RAM or use cloud deployment

### Issue: Request timeout from APIs

Solution:
- Check internet connection
- Wait a few minutes (API rate limiting)
- Use alternative data sources
- Reduce number of results requested

---

## Future Enhancements

- Drug interaction checker for multiple compounds
- ML model for drug target prediction
- Google Patents API integration for prior art search
- SMILES parser for chemical structure queries
- Multi-language support for international papers
- R&D cost and timeline estimation
- Cloud deployment (AWS Lambda, Google Cloud Run)
- User authentication for enterprise use
- Advanced filtering and sorting
- Batch query processing
- Custom knowledge base upload
- Export results to PDF/Excel
- API endpoint for programmatic access

---

## Resources

- PubMed API: https://www.ncbi.nlm.nih.gov/home/develop/api/
- ClinicalTrials.gov API: https://clinicaltrials.gov/api/gui
- Streamlit: https://docs.streamlit.io/
- ChromaDB: https://docs.trychroma.com/
- Sentence-Transformers: https://www.sbert.net/
- OpenAI API: https://platform.openai.com/docs/api-reference

---

## License

This project is created for educational and research purposes. Licensed under MIT License.

---

## Contributing

Have improvements? Feel free to fork and submit PRs!

---

## Contact & Support

For questions or support, please open an issue on GitHub.

Created for Jubilant Pharmova AI Internship Application

Last Updated: October 2025
