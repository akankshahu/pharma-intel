# Phase 2 Quick: Fast Embedding Creation (Simplified)
# ==================================================

import pandas as pd
import chromadb
import logging
import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_DB_PATH, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_embeddings():
    """
    Create sample embeddings quickly using random vectors for demonstration.
    This allows us to move forward with testing the RAG pipeline.
    """

    logger.info("=" * 60)
    logger.info("🚀 Creating Sample Knowledge Base (Phase 2 Quick)")
    logger.info("=" * 60)

    # Initialize ChromaDB
    logger.info(f"\n💾 Initializing ChromaDB at {CHROMA_DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Create collections
    pubmed_collection = chroma_client.get_or_create_collection(
        name="pubmed_abstracts",
        metadata={"hnsw:space": "cosine"}
    )

    clinical_collection = chroma_client.get_or_create_collection(
        name="clinical_trials",
        metadata={"hnsw:space": "cosine"}
    )

    logger.info("✅ Collections ready!")

    # Load PubMed data
    logger.info("\n📄 Processing PubMed articles...")
    try:
        df_pubmed = pd.read_csv('Phase_1_Data_Collection/data/pubmed_data.csv').head(15)

        for idx, row in df_pubmed.iterrows():
            # Create random embedding (demonstration)
            embedding = [random.random() for _ in range(384)]

            pubmed_collection.add(
                ids=[f"pubmed_{idx}"],
                embeddings=[embedding],
                documents=[str(row['abstract'])[:500] if pd.notna(row['abstract']) else "No abstract"],
                metadatas=[{
                    'pmid': str(row['pmid']),
                    'title': str(row['title'])[:200],
                    'url': str(row['url']),
                    'source': 'PubMed'
                }]
            )

        logger.info(f"✅ Added {len(df_pubmed)} PubMed documents")

    except Exception as e:
        logger.error(f"Error processing PubMed: {e}")

    # Load Clinical Trials data
    logger.info("🏥 Processing clinical trials...")
    try:
        df_trials = pd.read_csv('Phase_1_Data_Collection/data/clinical_trials_data.csv').head(30)

        for idx, row in df_trials.iterrows():
            # Create random embedding (demonstration)
            embedding = [random.random() for _ in range(384)]

            trial_text = f"{row['title']} - {row['condition']} - {row['status']}"

            clinical_collection.add(
                ids=[f"trial_{idx}"],
                embeddings=[embedding],
                documents=[trial_text],
                metadatas=[{
                    'nct_id': str(row['nct_id']),
                    'title': str(row['title'])[:200],
                    'condition': str(row['condition']),
                    'url': str(row['url']),
                    'source': 'ClinicalTrials.gov'
                }]
            )

        logger.info(f"✅ Added {len(df_trials)} clinical trial documents")

    except Exception as e:
        logger.error(f"Error processing trials: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Sample Knowledge Base Created!")
    logger.info("=" * 60)
    logger.info(f"💾 ChromaDB location: {CHROMA_DB_PATH}")
    logger.info("Ready for RAG pipeline testing!")
    logger.info("=" * 60)

if __name__ == "__main__":
    create_sample_embeddings()
