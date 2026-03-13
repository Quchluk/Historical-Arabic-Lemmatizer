#!/usr/bin/env python3
"""
CAMeL Tools Database Lemmatizer
================================
Lemmatizes Arabic texts from OpenITI database using CAMeL Tools.

CAMeL Tools is a superior alternative to Farasa:
- No external JAR files needed
- Better accuracy
- More features (morphology, POS, NER, etc.)
- Pure Python implementation
"""

import re
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import argparse
from tqdm import tqdm
import pandas as pd

# CAMeL Tools imports
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.word_tokenizer import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar

from lemmatizer.config import LOGS_DIR, DATA_DIR, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'camel_lemmatization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CAMeLLemmatizer:
    """Wrapper for CAMeL Tools lemmatization."""

    def __init__(self, dialect=None):
        """
        Initialize CAMeL Tools.

        Args:
            dialect: Database name (e.g., 'calima-msa-r13').
                     If None, uses the default open-source database.
        """
        logger.info(f"Initializing CAMeL Tools...")

        # Initialize morphology database
        logger.info("  - Loading morphology database...")
        if dialect:
            logger.info(f"    Using database: {dialect}")
            self.db = MorphologyDB.builtin_db(dialect)
        else:
            logger.info("    Using default open-source database")
            self.db = MorphologyDB.builtin_db()  # Uses default (no LDC license needed)

        # Initialize analyzer
        logger.info("  - Loading analyzer...")
        self.analyzer = Analyzer(self.db)

        # Initialize disambiguator (for better lemmatization)
        logger.info("  - Loading disambiguator...")
        try:
            if dialect and 'msa' in dialect:
                self.disambiguator = MLEDisambiguator.pretrained('calima-msa-r13')
            elif dialect and 'egy' in dialect:
                self.disambiguator = MLEDisambiguator.pretrained('calima-egy-r13')
            else:
                self.disambiguator = MLEDisambiguator.pretrained()  # default
        except Exception as e:
            logger.warning(f"    Disambiguator not available: {e}")
            self.disambiguator = None


        logger.info("✓ CAMeL Tools initialized")

    def lemmatize_text(self, text):
        """
        Lemmatize Arabic text and return word-lemma pairs.

        Args:
            text: Input Arabic text

        Returns:
            List of dicts with 'word' and 'lemma' keys
        """
        # Tokenize (CAMeL Tools handles normalization internally)
        tokens = simple_word_tokenize(text)

        results = []

        # Use disambiguator if available, otherwise use analyzer directly
        if self.disambiguator:
            # Disambiguate (get best analysis for each word)
            disambiguated = self.disambiguator.disambiguate(tokens)

            # Extract lemmas
            for word_obj in disambiguated:
                word = word_obj.word

                # Get lemma (or use word if no analysis available)
                if word_obj.analyses:
                    lemma = word_obj.analyses[0].analysis.get('lex', word)
                    pos = word_obj.analyses[0].analysis.get('pos', 'UNKNOWN')
                else:
                    lemma = word
                    pos = 'UNKNOWN'

                results.append({
                    'word': word,
                    'lemma': lemma,
                    'pos': pos
                })
        else:
            # Fall back to analyzer (without disambiguation)
            for word in tokens:
                analyses = self.analyzer.analyze(word)
                if analyses:
                    lemma = analyses[0].get('lex', word)
                    pos = analyses[0].get('pos', 'UNKNOWN')
                else:
                    lemma = word
                    pos = 'UNKNOWN'

                results.append({
                    'word': word,
                    'lemma': lemma,
                    'pos': pos
                })

        return results


class OpenITITextExtractor:
    """Extract clean text from OpenITI formatted files."""

    @staticmethod
    def is_arabic(text):
        """Check if text contains Arabic characters."""
        return bool(re.search(r'[\u0600-\u06FF]', text))

    @staticmethod
    def clean_openiti_line(line):
        """Remove OpenITI markup from a line."""
        # Skip metadata lines
        if line.startswith('#META#') or line.startswith('######'):
            return None

        # Remove structural markers
        line = re.sub(r'#+ ', '', line)  # Remove heading markers
        line = re.sub(r'PageV\d+P\d+', '', line)  # Remove page markers
        line = re.sub(r'ms\d+', '', line)  # Remove manuscript markers
        line = re.sub(r'~~', '', line)  # Remove poetry markers
        line = re.sub(r'@\w+', '', line)  # Remove tags

        # Clean up
        line = line.strip()

        return line if line and OpenITITextExtractor.is_arabic(line) else None

    @staticmethod
    def extract_text(file_path, max_chars=50000):
        """
        Extract clean Arabic text from OpenITI file.

        Args:
            file_path: Path to text file
            max_chars: Maximum characters to extract (for memory efficiency)
        """
        text_lines = []
        total_chars = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if total_chars >= max_chars:
                        break

                    cleaned = OpenITITextExtractor.clean_openiti_line(line)
                    if cleaned:
                        text_lines.append(cleaned)
                        total_chars += len(cleaned)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""

        return ' '.join(text_lines)


class DatabaseLemmatizer:
    """Main class for lemmatizing the entire database."""

    def __init__(self, db_path=None, output_dir=None, dialect='msa'):
        self.db_path = Path(db_path) if db_path else DATA_DIR / "db/db"
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "camel_lemmatized"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CAMeL Tools
        self.lemmatizer = CAMeLLemmatizer(dialect=dialect)

        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'total_words': 0,
            'unique_lemmas': Counter(),
            'lemma_to_words': defaultdict(set),
            'start_time': datetime.now()
        }

        # Progress tracking
        self.progress_file = self.output_dir / 'progress.json'
        self.processed_files = self.load_progress()

    def load_progress(self):
        """Load progress from previous runs."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Resuming from checkpoint: {len(data)} files already processed")
                return set(data)
        return set()

    def save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.processed_files), f, indent=2, ensure_ascii=False)

    def find_text_files(self):
        """Find all OpenITI text files in database."""
        logger.info("Scanning database for text files...")

        text_files = []

        # Iterate through period directories
        for period_dir in sorted(self.db_path.glob('*AH')):
            if not period_dir.is_dir():
                continue

            data_dir = period_dir / 'data'
            if not data_dir.exists():
                continue

            # Iterate through author directories
            for author_dir in data_dir.iterdir():
                if not author_dir.is_dir():
                    continue

                # Find text files (ending with -ara1, -ara2, etc.)
                for file_path in author_dir.rglob('*-ara*'):
                    if file_path.is_file() and not file_path.name.endswith('.yml'):
                        text_files.append(file_path)

        logger.info(f"Found {len(text_files)} text files")
        return text_files

    def lemmatize_file(self, file_path):
        """Lemmatize a single file and save results."""

        # Skip if already processed
        file_key = str(file_path.relative_to(self.db_path))
        if file_key in self.processed_files:
            return None

        try:
            # Extract text
            text = OpenITITextExtractor.extract_text(file_path, max_chars=100000)

            if not text or len(text) < 10:
                logger.warning(f"Skipping {file_path.name}: No valid text")
                self.stats['files_failed'] += 1
                return None

            # Lemmatize
            results = self.lemmatizer.lemmatize_text(text)

            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_words'] += len(results)

            for item in results:
                lemma = item['lemma']
                word = item['word']
                self.stats['unique_lemmas'][lemma] += 1
                self.stats['lemma_to_words'][lemma].add(word)

            # Save lemmatized output
            output_file = self.output_dir / f"{file_path.stem}_lemmatized.json"
            output_data = {
                'source_file': str(file_path),
                'processed_at': datetime.now().isoformat(),
                'num_words': len(results),
                'words_and_lemmas': results
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            # Mark as processed
            self.processed_files.add(file_key)

            return results

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            self.stats['files_failed'] += 1
            return None

    def process_database(self, limit=None):
        """Process entire database."""

        logger.info("="*80)
        logger.info("CAMeL TOOLS DATABASE LEMMATIZATION")
        logger.info("="*80)

        # Find all text files
        text_files = self.find_text_files()

        if limit:
            text_files = text_files[:limit]
            logger.info(f"Processing first {limit} files (limit)")

        # Process files with progress bar
        logger.info(f"\nProcessing {len(text_files)} files...")

        for file_path in tqdm(text_files, desc="Lemmatizing"):
            self.lemmatize_file(file_path)

            # Save progress periodically
            if self.stats['files_processed'] % 50 == 0:
                self.save_progress()
                self.save_lemma_registry()

        # Final save
        self.save_progress()
        self.save_lemma_registry()

        # Generate report
        self.generate_report()

    def save_lemma_registry(self):
        """Save lemma registry (similar to your existing registry format)."""

        # Create comprehensive registry
        registry_data = []

        for lemma, words in self.stats['lemma_to_words'].items():
            frequency = self.stats['unique_lemmas'][lemma]
            num_forms = len(words)

            registry_data.append({
                'lemma': lemma,
                'frequency': frequency,
                'num_forms': num_forms,
                'word_forms': sorted(list(words))
            })

        # Save as JSON
        registry_file = self.output_dir / 'lemma_registry.json'
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, ensure_ascii=False, indent=2)

        # Also save as CSV
        df = pd.DataFrame([
            {
                'lemma': item['lemma'],
                'frequency': item['frequency'],
                'num_forms': item['num_forms'],
                'word_forms': ', '.join(item['word_forms'][:10])  # First 10 forms
            }
            for item in registry_data
        ])

        csv_file = self.output_dir / 'lemma_registry.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        logger.info(f"✓ Lemma registry saved: {registry_file} and {csv_file}")

    def generate_report(self):
        """Generate summary report."""

        elapsed_time = datetime.now() - self.stats['start_time']

        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)Следующий шаг чтобы понять что происходит ассистенту ассистента чтоб ты мог сделать ты заводишь водишь выбираешь документы вот эти которые ты хочешь и нажимаешь анализировать и он этот прогонять через сети документ Siri Siri похуй ебать ты знал как хуёво вообще
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Files processed:    {self.stats['files_processed']:,}")
        logger.info(f"  Files failed:       {self.stats['files_failed']:,}")
        logger.info(f"  Total words:        {self.stats['total_words']:,}")
        logger.info(f"  Unique lemmas:      {len(self.stats['unique_lemmas']):,}")
        logger.info(f"  Elapsed time:       {elapsed_time}")

        # Top lemmas
        logger.info(f"\n🔝 Top 20 Most Frequent Lemmas:")
        for lemma, count in self.stats['unique_lemmas'].most_common(20):
            num_forms = len(self.stats['lemma_to_words'][lemma])
            logger.info(f"  {lemma:<20} {count:>6,} occurrences  ({num_forms} forms)")

        # Save summary
        summary = {
            'files_processed': self.stats['files_processed'],
            'files_failed': self.stats['files_failed'],
            'total_words': self.stats['total_words'],
            'unique_lemmas': len(self.stats['unique_lemmas']),
            'elapsed_time': str(elapsed_time),
            'completed_at': datetime.now().isoformat()
        }

        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n✓ Summary saved to: {summary_file}")
        logger.info(f"✓ Lemmatized files saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Lemmatize OpenITI database with CAMeL Tools')
    parser.add_argument('--db', default=None, help='Database directory path')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of files to process (for testing)')
    parser.add_argument('--dialect', default=None,
                       help='Arabic morphology database (e.g., calima-msa-r13). Default: uses open-source database.')

    args = parser.parse_args()

    # Create lemmatizer
    lemmatizer = DatabaseLemmatizer(
        db_path=args.db,
        output_dir=args.output,
        dialect=args.dialect
    )

    # Process database
    lemmatizer.process_database(limit=args.limit)


if __name__ == "__main__":
    main()

