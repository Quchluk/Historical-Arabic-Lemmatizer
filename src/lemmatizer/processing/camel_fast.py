#!/usr/bin/env python3
"""
Fast CAMeL Tools Database Lemmatizer
=====================================
Optimized version with:
- Multiprocessing (parallel workers)
- Batch processing
- Better progress tracking
- Skips already processed files
"""

import re
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from lemmatizer.config import LOGS_DIR, DATA_DIR, PROCESSED_DATA_DIR

# Setup logging with real-time output
class FlushHandler(logging.StreamHandler):
    """Handler that flushes after every emit"""
    def emit(self, record):
        super().emit(record)
        self.flush()

# Create handlers
file_handler = logging.FileHandler(LOGS_DIR / 'camel_lemmatization_fast.log')
console_handler = FlushHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


# Global variables for worker processes
_lemmatizer = None
_dialect = None


def init_worker(dialect):
    """Initialize CAMeL Tools in each worker process"""
    global _lemmatizer, _dialect
    _dialect = dialect

    # Import here to avoid issues with multiprocessing
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.word import simple_word_tokenize

    # Store in global for reuse
    _lemmatizer = {
        'simple_word_tokenize': simple_word_tokenize
    }

    # Initialize database
    if dialect:
        db = MorphologyDB.builtin_db(dialect)
    else:
        db = MorphologyDB.builtin_db()

    _lemmatizer['analyzer'] = Analyzer(db)

    # Initialize disambiguator
    try:
        if dialect and 'msa' in dialect:
            _lemmatizer['disambiguator'] = MLEDisambiguator.pretrained('calima-msa-r13')
        elif dialect and 'egy' in dialect:
            _lemmatizer['disambiguator'] = MLEDisambiguator.pretrained('calima-egy-r13')
        else:
            _lemmatizer['disambiguator'] = MLEDisambiguator.pretrained()
    except Exception:
        _lemmatizer['disambiguator'] = None


def is_arabic(text):
    """Check if text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))


def clean_openiti_line(line):
    """Remove OpenITI markup from a line."""
    if line.startswith('#META#') or line.startswith('######'):
        return None

    line = re.sub(r'#+ ', '', line)
    line = re.sub(r'PageV\d+P\d+', '', line)
    line = re.sub(r'ms\d+', '', line)
    line = re.sub(r'~~', '', line)
    line = re.sub(r'@\w+', '', line)
    line = line.strip()

    return line if line and is_arabic(line) else None


def extract_text(file_path, max_chars=100000):
    """Extract clean Arabic text from OpenITI file."""
    text_lines = []
    total_chars = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if total_chars >= max_chars:
                    break
                cleaned = clean_openiti_line(line)
                if cleaned:
                    text_lines.append(cleaned)
                    total_chars += len(cleaned)
    except Exception as e:
        return ""

    return ' '.join(text_lines)


def lemmatize_text(text):
    """Lemmatize Arabic text using global lemmatizer."""
    global _lemmatizer

    tokens = _lemmatizer['simple_word_tokenize'](text)
    results = []

    disambiguator = _lemmatizer.get('disambiguator')

    if disambiguator:
        disambiguated = disambiguator.disambiguate(tokens)
        for word_obj in disambiguated:
            word = word_obj.word
            if word_obj.analyses:
                lemma = word_obj.analyses[0].analysis.get('lex', word)
                pos = word_obj.analyses[0].analysis.get('pos', 'UNKNOWN')
            else:
                lemma = word
                pos = 'UNKNOWN'
            results.append({'word': word, 'lemma': lemma, 'pos': pos})
    else:
        analyzer = _lemmatizer['analyzer']
        for word in tokens:
            analyses = analyzer.analyze(word)
            if analyses:
                lemma = analyses[0].get('lex', word)
                pos = analyses[0].get('pos', 'UNKNOWN')
            else:
                lemma = word
                pos = 'UNKNOWN'
            results.append({'word': word, 'lemma': lemma, 'pos': pos})

    return results


def process_single_file(args):
    """Process a single file - called by worker processes."""
    file_path, output_dir, db_path = args

    try:
        # Extract text
        text = extract_text(file_path, max_chars=100000)

        if not text or len(text) < 10:
            return {'status': 'skipped', 'file': str(file_path), 'reason': 'no_text'}

        # Lemmatize
        results = lemmatize_text(text)

        # Save output - compact JSON for speed (no indent)
        output_file = Path(output_dir) / f"{Path(file_path).stem}_lemmatized.json"
        output_data = {
            'source_file': str(file_path),
            'processed_at': datetime.now().isoformat(),
            'num_words': len(results),
            'words_and_lemmas': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False)  # No indent = faster

        # Return stats
        file_key = str(Path(file_path).relative_to(db_path))
        return {
            'status': 'success',
            'file_key': file_key,
            'num_words': len(results),
            'lemmas': Counter(r['lemma'] for r in results)
        }

    except Exception as e:
        return {'status': 'error', 'file': str(file_path), 'error': str(e)}


class FastDatabaseLemmatizer:
    """Fast parallel lemmatizer for the entire database."""

    def __init__(self, db_path=None, output_dir=None, dialect='calima-msa-r13', num_workers=None):
        self.db_path = Path(db_path) if db_path else DATA_DIR / "db/db"
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR / "camel_lemmatized"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dialect = dialect
        self.num_workers = num_workers or mp.cpu_count()

        logger.info(f"System has {mp.cpu_count()} CPU cores")
        logger.info(f"Using {self.num_workers} parallel workers")

        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'files_skipped': 0,
            'total_words': 0,
            'unique_lemmas': Counter(),
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
                logger.info(f"✓ Resuming: {len(data)} files already processed")
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

        for period_dir in sorted(self.db_path.glob('*AH')):
            if not period_dir.is_dir():
                continue
            data_dir = period_dir / 'data'
            if not data_dir.exists():
                continue

            for author_dir in data_dir.iterdir():
                if not author_dir.is_dir():
                    continue
                for file_path in author_dir.rglob('*-ara*'):
                    if file_path.is_file() and not file_path.name.endswith('.yml'):
                        text_files.append(file_path)

        logger.info(f"Found {len(text_files)} total text files")
        return text_files

    def filter_unprocessed(self, text_files):
        """Filter out already processed files."""
        unprocessed = []
        for fp in text_files:
            file_key = str(fp.relative_to(self.db_path))
            if file_key not in self.processed_files:
                unprocessed.append(fp)

        skipped = len(text_files) - len(unprocessed)
        if skipped > 0:
            logger.info(f"✓ Skipping {skipped} already processed files")

        return unprocessed

    def process_database(self, limit=None):
        """Process entire database with parallel workers."""

        logger.info("="*80)
        logger.info("FAST CAMeL LEMMATIZER (Parallel Processing)")
        logger.info("="*80)
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Dialect: {self.dialect or 'default'}")

        # Find and filter files
        all_files = self.find_text_files()
        text_files = self.filter_unprocessed(all_files)

        if limit:
            text_files = text_files[:limit]
            logger.info(f"Limited to {limit} files")

        if not text_files:
            logger.info("✓ All files already processed!")
            return

        logger.info(f"\nProcessing {len(text_files)} files with {self.num_workers} workers...")

        # Prepare args for workers
        args_list = [(str(fp), str(self.output_dir), str(self.db_path)) for fp in text_files]

        # Process with parallel workers
        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=init_worker,
            initargs=(self.dialect,)
        ) as executor:

            futures = {executor.submit(process_single_file, args): args[0] for args in args_list}

            with tqdm(total=len(futures), desc="Lemmatizing") as pbar:
                for future in as_completed(futures):
                    result = future.result()

                    if result['status'] == 'success':
                        self.stats['files_processed'] += 1
                        self.stats['total_words'] += result['num_words']
                        self.stats['unique_lemmas'].update(result['lemmas'])
                        self.processed_files.add(result['file_key'])
                    elif result['status'] == 'skipped':
                        self.stats['files_skipped'] += 1
                    else:
                        self.stats['files_failed'] += 1
                        logger.warning(f"Error: {result.get('error', 'unknown')}")

                    pbar.update(1)

                    # Save progress every 100 files
                    if (self.stats['files_processed'] + self.stats['files_skipped']) % 100 == 0:
                        self.save_progress()

        # Final save
        self.save_progress()
        self.save_lemma_registry()
        self.generate_report()

    def save_lemma_registry(self):
        """Save lemma registry."""
        registry_data = [
            {'lemma': lemma, 'frequency': count}
            for lemma, count in self.stats['unique_lemmas'].most_common()
        ]

        registry_file = self.output_dir / 'lemma_registry.json'
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, ensure_ascii=False, indent=2)

        df = pd.DataFrame(registry_data)
        csv_file = self.output_dir / 'lemma_registry.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        logger.info(f"✓ Lemma registry saved")

    def generate_report(self):
        """Generate summary report."""
        elapsed = datetime.now() - self.stats['start_time']

        logger.info("\n" + "="*80)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"  Files processed:  {self.stats['files_processed']:,}")
        logger.info(f"  Files skipped:    {self.stats['files_skipped']:,}")
        logger.info(f"  Files failed:     {self.stats['files_failed']:,}")
        logger.info(f"  Total words:      {self.stats['total_words']:,}")
        logger.info(f"  Unique lemmas:    {len(self.stats['unique_lemmas']):,}")
        logger.info(f"  Elapsed time:     {elapsed}")

        if self.stats['files_processed'] > 0:
            rate = self.stats['files_processed'] / elapsed.total_seconds()
            logger.info(f"  Processing rate:  {rate:.2f} files/sec")

        # Save summary
        summary = {
            'files_processed': self.stats['files_processed'],
            'files_skipped': self.stats['files_skipped'],
            'files_failed': self.stats['files_failed'],
            'total_words': self.stats['total_words'],
            'unique_lemmas': len(self.stats['unique_lemmas']),
            'elapsed_time': str(elapsed),
            'completed_at': datetime.now().isoformat()
        }

        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Fast parallel CAMeL lemmatizer')
    parser.add_argument('--db', default=None, help='Database directory')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit files to process')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--dialect', default='calima-msa-r13', help='Dialect database')

    args = parser.parse_args()

    lemmatizer = FastDatabaseLemmatizer(
        db_path=args.db,
        output_dir=args.output,
        dialect=args.dialect,
        num_workers=args.workers
    )

    lemmatizer.process_database(limit=args.limit)


if __name__ == "__main__":
    main()

