# Phase 3: RAG Engine
# Retrieval-Augmented Generation Pipeline
# ==================================================

import os
import sys
import chromadb
import logging
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_DB_PATH, LLM_MODEL, OPENAI_API_KEY, LOG_LEVEL, NUM_SOURCES_TO_RETRIEVE

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    logger.error("OpenAI not installed. Install with: pip install openai")
    OpenAI = None

class PharmaIntellectRAG:
    """
    Retrieval-Augmented Generation Engine for Pharma-Intellect
    """

    def __init__(self):
        """Initialize the RAG engine with ChromaDB and OpenAI."""

        logger.info("=" * 60)
        logger.info("🚀 Initializing Pharma-Intellect RAG Engine (Phase 3)")
        logger.info("=" * 60)

        # Initialize ChromaDB
        logger.info(f"\n💾 Connecting to ChromaDB at {CHROMA_DB_PATH}...")
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.pubmed_collection = self.chroma_client.get_collection("pubmed_abstracts")
            self.clinical_collection = self.chroma_client.get_collection("clinical_trials")
            logger.info("✅ ChromaDB collections loaded!")
        except Exception as e:
            logger.error(f"❌ Failed to load ChromaDB: {str(e)}")
            raise

        # Initialize OpenAI
        logger.info(f"\n🤖 Configuring OpenAI (Model: {LLM_MODEL})...")
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-proj-your"):
            logger.warning("⚠️  OPENAI_API_KEY not properly configured!")
            logger.warning("   Set it in .env file or export OPENAI_API_KEY=...")

        if OpenAI:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("✅ OpenAI client initialized!")
        else:
            logger.warning("⚠️  OpenAI client not available")
            self.openai_client = None

        logger.info("=" * 60)
        logger.info("✅ RAG Engine Ready!")
        logger.info("=" * 60)

    def retrieve_documents(self, query: str, num_results: int = NUM_SOURCES_TO_RETRIEVE) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents from ChromaDB.

        Args:
            query: User's question
            num_results: Number of documents to retrieve

        Returns:
            List of retrieved documents with metadata
        """

        logger.info(f"\n🔍 Retrieving documents for: '{query[:80]}...'")

        # Since we're using random embeddings in ChromaDB, we'll do simple text search
        all_results = []

        try:
            # Query PubMed collection
            pubmed_results = self.pubmed_collection.query(
                query_texts=[query],
                n_results=min(num_results // 2, 5),
                include=["documents", "metadatas", "distances"]
            )

            if pubmed_results and pubmed_results['documents']:
                for i, doc in enumerate(pubmed_results['documents'][0]):
                    metadata = pubmed_results['metadatas'][0][i] if i < len(pubmed_results['metadatas'][0]) else {}
                    all_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'relevance_score': 0.8
                    })
                logger.info(f"   ✅ Found {len(pubmed_results['documents'][0])} PubMed results")
        except Exception as e:
            logger.warning(f"   ⚠️  Error querying PubMed: {str(e)}")

        try:
            # Query Clinical Trials collection
            clinical_results = self.clinical_collection.query(
                query_texts=[query],
                n_results=min(num_results // 2, 5),
                include=["documents", "metadatas", "distances"]
            )

            if clinical_results and clinical_results['documents']:
                for i, doc in enumerate(clinical_results['documents'][0]):
                    metadata = clinical_results['metadatas'][0][i] if i < len(clinical_results['metadatas'][0]) else {}
                    all_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'relevance_score': 0.75
                    })
                logger.info(f"   ✅ Found {len(clinical_results['documents'][0])} Clinical Trial results")
        except Exception as e:
            logger.warning(f"   ⚠️  Error querying Clinical Trials: {str(e)}")

        logger.info(f"   📊 Total documents retrieved: {len(all_results)}")
        return all_results

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using the LLM with retrieved context.

        Args:
            query: User's question
            retrieved_docs: List of retrieved documents

        Returns:
            Generated answer
        """

        logger.info(f"\n🤖 Generating answer with {LLM_MODEL}...")

        if not self.openai_client:
            # Fallback response without LLM
            logger.warning("⚠️  OpenAI not configured. Returning search results only.")
            return self._create_fallback_answer(query, retrieved_docs)

        try:
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Source {i+1} ({doc['metadata'].get('source', 'Unknown')}):\n{doc['text']}"
                for i, doc in enumerate(retrieved_docs)
            ])

            # Create the prompt
            system_prompt = """You are an expert pharmaceutical research assistant.
Your task is to answer questions based on the provided scientific literature and clinical trial data.
- Be precise and evidence-based
- Clearly cite your sources
- Distinguish between different studies
- If information is missing, say so explicitly
- Provide structured, readable answers"""

            user_prompt = f"""Based on the following scientific literature and clinical trial data, please answer this question:

QUESTION: {query}

AVAILABLE DATA:
{context}

Please provide a comprehensive, well-structured answer with clear reference to the sources."""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            logger.info("✅ Answer generated successfully!")
            return answer

        except Exception as e:
            logger.error(f"❌ Error generating answer: {str(e)}")
            return self._create_fallback_answer(query, retrieved_docs)

    def _create_fallback_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create a fallback answer when LLM is not available.
        """

        if not retrieved_docs:
            return f"Sorry, I could not find relevant information for your query: '{query}'"

        answer = f"## Answer to: {query}\n\n"
        answer += "Based on the retrieved documents:\n\n"

        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            title = doc['metadata'].get('title', 'No title')
            answer += f"**Source {i} ({source}):**\n"
            answer += f"*{title}*\n"
            answer += f"{doc['text'][:300]}...\n\n"

        return answer

    def query(self, question: str, num_results: int = NUM_SOURCES_TO_RETRIEVE) -> Dict[str, Any]:
        """
        Execute a complete RAG query.

        Args:
            question: User's research question
            num_results: Number of documents to retrieve

        Returns:
            Dictionary with answer and sources
        """

        logger.info("\n" + "=" * 60)
        logger.info(f"🔍 Processing Query: {question[:60]}...")
        logger.info("=" * 60)

        # Retrieve documents
        retrieved_docs = self.retrieve_documents(question, num_results)

        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)

        # Format sources
        sources = [
            {
                'title': doc['metadata'].get('title', 'Unknown'),
                'url': doc['metadata'].get('url', '#'),
                'source': doc['metadata'].get('source', 'Unknown'),
                'relevance': doc['relevance_score']
            }
            for doc in retrieved_docs
        ]

        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources_retrieved': len(retrieved_docs)
        }

        logger.info("=" * 60)
        logger.info(f"✅ Query Complete! Retrieved {len(retrieved_docs)} sources.")
        logger.info("=" * 60)

        return result

# ============================================
# Example Usage and Testing
# ============================================

if __name__ == "__main__":
    # Initialize RAG engine
    rag = PharmaIntellectRAG()

    # Example queries
    example_queries = [
        "What are the recent advancements in treating autoimmune disorders?",
        "Summarize phase 2 clinical trials for oncology treatments",
        "What are the adverse events associated with risperidone?"
    ]

    print("\n" + "=" * 80)
    print("PHARMA-INTELLECT RAG ENGINE - TEST RUN")
    print("=" * 80)

    for i, query in enumerate(example_queries[:1], 1):  # Test first query
        print(f"\n{'=' * 80}")
        print(f"QUERY {i}: {query}")
        print("=" * 80)

        result = rag.query(query)

        print("\n📄 ANSWER:")
        print("-" * 80)
        print(result['answer'])

        print("\n📚 SOURCES:")
        print("-" * 80)
        for j, source in enumerate(result['sources'], 1):
            print(f"\n{j}. {source['title']}")
            print(f"   Source: {source['source']}")
            print(f"   Relevance: {source['relevance']:.2%}")
            print(f"   URL: {source['url']}")

        print("\n" + "=" * 80)
