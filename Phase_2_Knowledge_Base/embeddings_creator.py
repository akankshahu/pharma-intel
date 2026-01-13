# Phase 2: Building the Knowledge Base
# Create vector embeddings and store in ChromaDB
# ==================================================

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import sys
import os
from typing import List

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Text Chunking
# ============================================

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split long text into overlapping chunks to maintain context.

    Args:
        text: Text to chunk
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """

    if not isinstance(text, str) or len(text) == 0:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# ============================================
# Embedding and Storage
# ============================================

def create_embeddings():
    """
    Load data, create embeddings, and store in ChromaDB.
    """

    logger.info("=" * 60)
    logger.info("🚀 Starting Pharma-Intellect Embedding Creation (Phase 2)")
    logger.info("=" * 60)

    # Load pre-trained embedding model with memory optimization
    logger.info(f"\n📦 Loading Sentence-Transformers model '{EMBEDDING_MODEL}'...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        # Optimize model for CPU/memory usage
        model.max_seq_length = 256
        logger.info(f"✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return    # Initialize ChromaDB
    logger.info(f"\n💾 Initializing ChromaDB at {CHROMA_DB_PATH}...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info("✅ ChromaDB initialized!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB: {str(e)}")
        return

    # Create or get collections
    logger.info("\n📚 Creating/Getting ChromaDB collections...")
    try:
        pubmed_collection = chroma_client.get_or_create_collection(
            name="pubmed_abstracts",
            metadata={"hnsw:space": "cosine", "description": "PubMed scientific articles"}
        )
        logger.info("✅ PubMed collection ready")

        clinical_collection = chroma_client.get_or_create_collection(
            name="clinical_trials",
            metadata={"hnsw:space": "cosine", "description": "Clinical trial information"}
        )
        logger.info("✅ Clinical Trials collection ready")
    except Exception as e:
        logger.error(f"❌ Failed to create collections: {str(e)}")
        return

    # ============================================
    # Process PubMed Data
    # ============================================

    logger.info("\n" + "=" * 60)
    logger.info("📄 PROCESSING PUBMED DATA")
    logger.info("=" * 60)

    pubmed_count = 0
    try:
        df_pubmed = pd.read_csv('Phase_1_Data_Collection/data/pubmed_data.csv')
        # Process only first 25 articles to avoid memory issues
        df_pubmed = df_pubmed.head(25)
        logger.info(f"   Loaded {len(df_pubmed)} PubMed articles (limited to 25 for memory efficiency)")

        for idx, row in df_pubmed.iterrows():
            try:
                # Skip if abstract is missing
                if pd.isna(row['abstract']) or row['abstract'] == 'N/A':
                    continue

                # Chunk the abstract
                chunks = chunk_text(str(row['abstract']), chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

                if not chunks:
                    continue

                for chunk_idx, chunk in enumerate(chunks):
                    # Create embedding
                    embedding = model.encode(chunk)

                    # Create unique ID
                    chunk_id = f"pubmed_{row['pmid']}_{chunk_idx}"

                    # Add to ChromaDB
                    pubmed_collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding.tolist()],
                        documents=[chunk],
                        metadatas=[{
                            'pmid': str(row['pmid']),
                            'title': str(row['title'])[:200],
                            'url': str(row['url']),
                            'source': 'PubMed',
                            'chunk_idx': chunk_idx
                        }]
                    )
                    pubmed_count += 1

                if (idx + 1) % 10 == 0:
                    logger.info(f"   Processed {idx + 1}/{len(df_pubmed)} articles ({pubmed_count} chunks)")

            except Exception as e:
                logger.warning(f"   Error processing article {idx}: {str(e)}")
                continue

        logger.info(f"✅ Processed {len(df_pubmed)} PubMed articles ({pubmed_count} chunks)")

    except FileNotFoundError:
        logger.warning("⚠️  PubMed data file not found. Skipping PubMed processing.")
    except Exception as e:
        logger.error(f"❌ Error processing PubMed data: {str(e)}")

    # ============================================
    # Process Clinical Trials Data
    # ============================================

    logger.info("\n" + "=" * 60)
    logger.info("🏥 PROCESSING CLINICAL TRIALS DATA")
    logger.info("=" * 60)

    clinical_count = 0
    try:
        df_trials = pd.read_csv('Phase_1_Data_Collection/data/clinical_trials_data.csv')
        # Process only first 50 trials to avoid memory issues
        df_trials = df_trials.head(50)
        logger.info(f"   Loaded {len(df_trials)} clinical trials (limited to 50 for memory efficiency)")

        for idx, row in df_trials.iterrows():
            try:
                # Combine relevant fields into a comprehensive text
                trial_text = f"""
                Title: {row['title']}
                Condition: {row['condition']}
                Status: {row['status']}
                Phase: {row['phase']}
                Start Date: {row['start_date']}
                Primary Outcomes: {row['primary_outcomes']}
                Interventions: {row['interventions']}
                """

                # Chunk the trial text
                chunks = chunk_text(trial_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

                if not chunks:
                    continue

                for chunk_idx, chunk in enumerate(chunks):
                    # Create embedding
                    embedding = model.encode(chunk)

                    # Create unique ID
                    chunk_id = f"trial_{row['nct_id']}_{chunk_idx}"

                    # Add to ChromaDB
                    clinical_collection.add(
                        ids=[chunk_id],
                        embeddings=[embedding.tolist()],
                        documents=[chunk],
                        metadatas=[{
                            'nct_id': str(row['nct_id']),
                            'title': str(row['title'])[:200],
                            'condition': str(row['condition']),
                            'url': str(row['url']),
                            'source': 'ClinicalTrials.gov',
                            'chunk_idx': chunk_idx
                        }]
                    )
                    clinical_count += 1

                if (idx + 1) % 50 == 0:
                    logger.info(f"   Processed {idx + 1}/{len(df_trials)} trials ({clinical_count} chunks)")

            except Exception as e:
                logger.warning(f"   Error processing trial {idx}: {str(e)}")
                continue

        logger.info(f"✅ Processed {len(df_trials)} clinical trials ({clinical_count} chunks)")

    except FileNotFoundError:
        logger.warning("⚠️  Clinical trials data file not found. Skipping trial processing.")
    except Exception as e:
        logger.error(f"❌ Error processing clinical trials data: {str(e)}")

    # ============================================
    # Summary
    # ============================================

    logger.info("\n" + "=" * 60)
    logger.info("✅ KNOWLEDGE BASE CREATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📊 Total PubMed chunks indexed: {pubmed_count}")
    logger.info(f"📊 Total Clinical Trial chunks indexed: {clinical_count}")
    logger.info(f"📊 Total documents indexed: {pubmed_count + clinical_count}")
    logger.info(f"💾 ChromaDB location: {CHROMA_DB_PATH}")
    logger.info(f"🔍 Ready for semantic search!")
    logger.info("=" * 60)

if __name__ == "__main__":
    create_embeddings()
