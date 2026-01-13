# Phase 1: Alternative PubMed Data Collection
# Uses NCBI Europe API as fallback when main PubMed API fails
# ==================================================

import requests
import pandas as pd
import json
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_pubmed_via_europe_api(keywords: List[str], max_results: int = 30) -> List[Dict[str, Any]]:
    """
    Fetch articles from Europe PMC API as alternative to NCBI PubMed.

    Args:
        keywords: List of search terms
        max_results: Maximum results per keyword

    Returns:
        List of article dictionaries

    API Documentation: https://europepmc.org/api
    """

    articles = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Pharma-Intellect/1.0'})

    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    for keyword in keywords:
        try:
            logger.info(f"🔍 Searching Europe PMC for: '{keyword}'")

            params = {
                'query': keyword,
                'format': 'json',
                'pageSize': min(max_results, 100),
                'isOpenAccess': 'Y'
            }

            response = session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get('resultList', {}).get('result', [])
            total_results = data.get('hitCount', 0)

            logger.info(f"   Found {len(results)} articles (Total: {total_results}) for '{keyword}'")

            if not results:
                logger.warning(f"   ⚠️  No articles found for '{keyword}'")
                continue

            for article in results:
                try:
                    article_info = {
                        'pmid': article.get('pmid', article.get('id', 'N/A')),
                        'title': article.get('title', 'N/A'),
                        'abstract': article.get('abstractText', 'No abstract available'),
                        'pub_date': article.get('pubYear', 'N/A'),
                        'authors': article.get('authorList', {}).get('author', []),
                        'source': 'PubMed (Europe PMC)',
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid', article.get('id', ''))}/" if article.get('pmid') else f"https://europepmc.org/article/MED/{article.get('id', '')}",
                        'keyword': keyword,
                        'collected_date': datetime.now().isoformat()
                    }
                    articles.append(article_info)

                except Exception as e:
                    logger.warning(f"   Error parsing article: {str(e)}")
                    continue

            logger.info(f"   ✅ Successfully collected {len(results)} articles for '{keyword}'")

        except requests.exceptions.Timeout:
            logger.error(f"   ❌ Timeout while fetching data for '{keyword}'")
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Error fetching data for '{keyword}': {str(e)}")
            continue
        except Exception as e:
            logger.error(f"   ❌ Unexpected error for '{keyword}': {str(e)}")
            continue

    logger.info(f"📊 Total articles collected: {len(articles)}")
    return articles

def create_sample_pubmed_data() -> List[Dict[str, Any]]:
    """
    Create sample PubMed-like data for demonstration when API fails.
    """

    sample_articles = [
        {
            'pmid': '39214589',
            'title': 'Insights into targeted drug delivery systems for oncology',
            'abstract': 'Recent advances in targeted drug delivery have revolutionized cancer treatment. This review examines the latest developments in nanoparticle-based delivery systems, antibody-drug conjugates, and cell-targeting therapies.',
            'pub_date': '2024',
            'authors': ['Author A', 'Author B'],
            'source': 'PubMed (Sample)',
            'url': 'https://pubmed.ncbi.nlm.nih.gov/39214589/',
            'keyword': 'oncology drug discovery',
            'collected_date': datetime.now().isoformat()
        },
        {
            'pmid': '39181234',
            'title': 'Emerging therapies for autoimmune disorders: A systematic review',
            'abstract': 'This systematic review analyzes emerging immunomodulatory therapies for autoimmune diseases. We examine JAK inhibitors, TNF-alpha inhibitors, and novel biologics showing promise in clinical trials.',
            'pub_date': '2024',
            'authors': ['Author C', 'Author D'],
            'source': 'PubMed (Sample)',
            'url': 'https://pubmed.ncbi.nlm.nih.gov/39181234/',
            'keyword': 'autoimmune disorders treatment',
            'collected_date': datetime.now().isoformat()
        },
        {
            'pmid': '39145678',
            'title': 'Risperidone clinical efficacy in psychiatric disorders: Meta-analysis',
            'abstract': 'Meta-analysis of 45 randomized controlled trials examining risperidone efficacy across schizophrenia, bipolar disorder, and autism spectrum disorder. Results show robust antipsychotic effects with manageable side effect profiles.',
            'pub_date': '2023',
            'authors': ['Author E', 'Author F'],
            'source': 'PubMed (Sample)',
            'url': 'https://pubmed.ncbi.nlm.nih.gov/39145678/',
            'keyword': 'risperidone clinical trials',
            'collected_date': datetime.now().isoformat()
        },
        {
            'pmid': '39098765',
            'title': 'Drug repurposing strategies in oncology: Past successes and future prospects',
            'abstract': 'Drug repurposing has led to several FDA approvals in oncology. This review discusses computational approaches, biomarker-driven strategies, and successful case studies of repurposed cancer therapeutics.',
            'pub_date': '2024',
            'authors': ['Author G', 'Author H'],
            'source': 'PubMed (Sample)',
            'url': 'https://pubmed.ncbi.nlm.nih.gov/39098765/',
            'keyword': 'drug repurposing',
            'collected_date': datetime.now().isoformat()
        },
        {
            'pmid': '39056789',
            'title': 'Phase 2 clinical trials: Design and biomarker-driven approaches',
            'abstract': 'Phase 2 trials bridge the gap between preclinical and late-stage development. This paper reviews adaptive designs, basket trials, and the role of biomarkers in modern Phase 2 development strategies.',
            'pub_date': '2024',
            'authors': ['Author I', 'Author J'],
            'source': 'PubMed (Sample)',
            'url': 'https://pubmed.ncbi.nlm.nih.gov/39056789/',
            'keyword': 'phase 2 clinical trials',
            'collected_date': datetime.now().isoformat()
        },
    ]

    return sample_articles

def main():
    """
    Try to collect PubMed data from Europe PMC or use sample data.
    """

    logger.info("=" * 60)
    logger.info("📚 Attempting Alternative PubMed Data Collection")
    logger.info("=" * 60)

    keywords = [
        'oncology drug discovery',
        'autoimmune disorders treatment',
        'risperidone clinical trials',
        'drug repurposing',
        'phase 2 clinical trials'
    ]

    # Try Europe PMC API first
    articles = fetch_pubmed_via_europe_api(keywords, max_results=20)

    # If no results, use sample data
    if not articles:
        logger.warning("⚠️  No articles from Europe PMC. Using sample data for demonstration.")
        articles = create_sample_pubmed_data()

    # Save to CSV
    if articles:
        try:
            df = pd.DataFrame(articles)
            df = df.drop_duplicates(subset=['pmid'], keep='first')
            df.to_csv('Phase_1_Data_Collection/data/pubmed_data.csv', index=False, encoding='utf-8')
            logger.info(f"✅ Saved {len(df)} articles to Phase_1_Data_Collection/data/pubmed_data.csv")
        except Exception as e:
            logger.error(f"❌ Error saving data: {str(e)}")

    logger.info("=" * 60)

if __name__ == "__main__":
    main()
