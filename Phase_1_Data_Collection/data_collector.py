# Phase 1: Data Collection and ETL
# Collects data from PubMed and ClinicalTrials.gov APIs
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

from config import (
    PUBMED_BASE_URL,
    PUBMED_MAX_RESULTS,
    PUBMED_KEYWORDS,
    CLINICAL_TRIALS_BASE_URL,
    CLINICAL_TRIALS_MAX_RESULTS,
    CLINICAL_TRIALS_CONDITIONS,
    LOG_LEVEL
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# PubMed Data Collection
# ============================================

def fetch_pubmed_data(keywords: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch articles from PubMed API.

    Args:
        keywords: List of search terms (e.g., ["oncology", "autoimmune disorders"])
        max_results: Maximum number of articles to retrieve per keyword

    Returns:
        List of dictionaries with article metadata

    API Documentation: https://www.ncbi.nlm.nih.gov/home/develop/api/
    """

    articles = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Pharma-Intellect/1.0'})

    for keyword in keywords:
        try:
            logger.info(f"🔍 Searching PubMed for: '{keyword}'")

            # Step 1: Search for article IDs
            search_params = {
                'db': 'pubmed',
                'term': keyword,
                'retmax': max_results,
                'rettype': 'json'
            }

            search_url = f"{PUBMED_BASE_URL}esearch.fcgi"
            response = session.get(search_url, params=search_params, timeout=10)
            response.raise_for_status()
            data = response.json()

            id_list = data.get('esearchresult', {}).get('idlist', [])
            total_count = data.get('esearchresult', {}).get('count', '0')

            logger.info(f"   Found {len(id_list)} articles (Total: {total_count}) for '{keyword}'")

            if not id_list:
                logger.warning(f"   ⚠️  No articles found for '{keyword}'")
                continue

            # Step 2: Fetch details for each ID (batch fetching)
            ids_str = ",".join(id_list[:max_results])

            fetch_params = {
                'db': 'pubmed',
                'id': ids_str,
                'rettype': 'abstract',
                'retmode': 'json'
            }

            fetch_url = f"{PUBMED_BASE_URL}efetch.fcgi"
            fetch_response = session.get(fetch_url, params=fetch_params, timeout=15)
            fetch_response.raise_for_status()
            fetch_data = fetch_response.json()

            # Step 3: Parse and extract relevant fields
            pubmed_articles = fetch_data.get('result', {})

            for article_id in id_list[:max_results]:
                try:
                    if article_id not in pubmed_articles:
                        continue

                    article = pubmed_articles[article_id]

                    # Extract key information
                    article_info = {
                        'pmid': article.get('uid', 'N/A'),
                        'title': article.get('title', 'N/A'),
                        'abstract': article.get('abstract', 'N/A'),
                        'pub_date': article.get('pub_date', 'N/A'),
                        'authors': article.get('authors', []),
                        'source': 'PubMed',
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.get('uid', '')}/",
                        'keyword': keyword,
                        'collected_date': datetime.now().isoformat()
                    }
                    articles.append(article_info)

                except Exception as e:
                    logger.warning(f"   Error parsing article {article_id}: {str(e)}")
                    continue

            logger.info(f"   ✅ Successfully collected {len(id_list)} articles for '{keyword}'")

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

# ============================================
# ClinicalTrials.gov Data Collection
# ============================================

def fetch_clinical_trials_data(conditions: List[str], max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch clinical trial data from ClinicalTrials.gov API v2.

    Args:
        conditions: List of medical conditions to search for
        max_results: Maximum trials per condition

    Returns:
        List of dictionaries with trial metadata

    API Documentation: https://clinicaltrials.gov/api/gui
    """

    trials = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Pharma-Intellect/1.0'})

    for condition in conditions:
        try:
            logger.info(f"🔍 Searching ClinicalTrials.gov for: '{condition}'")

            # ClinicalTrials.gov v2 API query
            params = {
                'query.cond': condition,
                'pageSize': min(max_results, 100),  # API max is 100
                'fields': 'NCTId,BriefTitle,Condition,InterventionName,PrimaryOutcomeMeasure,OverallStatus,Phase,StartDate'
            }

            response = session.get(CLINICAL_TRIALS_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            studies = data.get('studies', [])
            total_count = data.get('totalCount', len(studies))

            logger.info(f"   Found {len(studies)} trials (Total: {total_count}) for '{condition}'")

            if not studies:
                logger.warning(f"   ⚠️  No trials found for '{condition}'")
                continue

            # Parse each study
            for study in studies:
                try:
                    protocol_section = study.get('protocolSection', {})
                    id_module = protocol_section.get('identificationModule', {})
                    status_module = protocol_section.get('statusModule', {})
                    design_module = protocol_section.get('designModule', {})
                    outcomes_module = protocol_section.get('outcomesModule', {})
                    interventions_module = protocol_section.get('interventionsModule', {})

                    # Extract outcomes
                    primary_outcomes = []
                    for outcome in outcomes_module.get('primaryOutcomes', []):
                        primary_outcomes.append(outcome.get('measure', 'N/A'))

                    # Extract interventions
                    interventions = []
                    for intervention in interventions_module.get('interventionList', []):
                        interventions.append(intervention.get('name', 'N/A'))

                    trial_info = {
                        'nct_id': id_module.get('nctId', 'N/A'),
                        'title': id_module.get('briefTitle', 'N/A'),
                        'condition': condition,
                        'status': status_module.get('overallStatus', 'N/A'),
                        'phase': design_module.get('phases', ['N/A'])[0] if design_module.get('phases') else 'N/A',
                        'start_date': status_module.get('startDateStruct', {}).get('date', 'N/A'),
                        'primary_outcomes': json.dumps(primary_outcomes),
                        'interventions': json.dumps(interventions),
                        'source': 'ClinicalTrials.gov',
                        'url': f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}",
                        'collected_date': datetime.now().isoformat()
                    }
                    trials.append(trial_info)

                except Exception as e:
                    logger.warning(f"   Error parsing trial: {str(e)}")
                    continue

            logger.info(f"   ✅ Successfully collected {len(studies)} trials for '{condition}'")

        except requests.exceptions.Timeout:
            logger.error(f"   ❌ Timeout while fetching trials for '{condition}'")
            continue
        except requests.exceptions.RequestException as e:
            logger.error(f"   ❌ Error fetching trials for '{condition}': {str(e)}")
            continue
        except Exception as e:
            logger.error(f"   ❌ Unexpected error for '{condition}': {str(e)}")
            continue

    logger.info(f"📊 Total clinical trials collected: {len(trials)}")
    return trials

# ============================================
# Data Processing and Storage
# ============================================

def save_data_to_csv(data, filename: str) -> None:
    """
    Save collected data to CSV file.

    Args:
        data: List of dictionaries or DataFrame to save
        filename: Output filename
    """

    try:
        # Handle both DataFrames and lists
        if isinstance(data, pd.DataFrame):
            df = data
            if df.empty:
                logger.warning(f"⚠️  No data to save to {filename}")
                return
        else:
            if not data:
                logger.warning(f"⚠️  No data to save to {filename}")
                return
            df = pd.DataFrame(data)

        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"✅ Saved {len(df)} records to {filename}")
    except Exception as e:
        logger.error(f"❌ Error saving data to {filename}: {str(e)}")

def clean_and_deduplicate(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """
    Clean and remove duplicate entries from dataframe.

    Args:
        df: DataFrame to clean
        id_column: Column name to check for duplicates

    Returns:
        Cleaned DataFrame
    """

    original_count = len(df)
    df_clean = df.drop_duplicates(subset=[id_column], keep='first')
    removed_count = original_count - len(df_clean)

    if removed_count > 0:
        logger.info(f"🧹 Removed {removed_count} duplicate records")

    # Remove rows with empty abstracts/titles
    df_clean = df_clean.dropna(subset=['title'])

    return df_clean

# ============================================
# Main Execution
# ============================================

def main():
    """
    Main function to orchestrate data collection.
    """

    logger.info("=" * 60)
    logger.info("🚀 Starting Pharma-Intellect Data Collection (Phase 1)")
    logger.info("=" * 60)

    # Collect PubMed data
    logger.info("\n📚 COLLECTING PUBMED DATA...")
    logger.info("-" * 60)
    pubmed_articles = fetch_pubmed_data(PUBMED_KEYWORDS, PUBMED_MAX_RESULTS)

    if pubmed_articles:
        df_pubmed = pd.DataFrame(pubmed_articles)
        df_pubmed = clean_and_deduplicate(df_pubmed, 'pmid')
        save_data_to_csv(df_pubmed, 'Phase_1_Data_Collection/data/pubmed_data.csv')

    # Collect ClinicalTrials.gov data
    logger.info("\n🏥 COLLECTING CLINICAL TRIALS DATA...")
    logger.info("-" * 60)
    clinical_trials = fetch_clinical_trials_data(CLINICAL_TRIALS_CONDITIONS, CLINICAL_TRIALS_MAX_RESULTS)

    if clinical_trials:
        df_trials = pd.DataFrame(clinical_trials)
        df_trials = clean_and_deduplicate(df_trials, 'nct_id')
        save_data_to_csv(df_trials, 'Phase_1_Data_Collection/data/clinical_trials_data.csv')

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Data Collection Complete!")
    logger.info("=" * 60)
    logger.info(f"📊 PubMed Articles: {len(pubmed_articles)}")
    logger.info(f"🏥 Clinical Trials: {len(clinical_trials)}")
    logger.info(f"📁 Data saved to: Phase_1_Data_Collection/data/")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
