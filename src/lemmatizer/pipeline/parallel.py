#!/usr/bin/env python3
"""
Parallel Lemmatization Pipeline
================================
Runs lemmatization continuously while validation/LLM/cleaning runs in parallel.

Architecture:
- Main thread: Lemmatization (runs non-stop)
- Background thread: Validation + LLM + Clean output (processes completed batches)

This ensures maximum throughput - lemmatization never waits for validation.
"""

import os
import sys

# Suppress tokenizer parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import logging
import argparse
import threading
import queue
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd
from lemmatizer.config import LOGS_DIR, DATA_DIR, PROCESSED_DATA_DIR

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging with real-time output
class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pipeline_parallel.log'),
        FlushHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


class ParallelLemmatizationPipeline:
    """
    Parallel pipeline that runs:
    - Lemmatization in the main thread (continuous)
    - Validation + LLM + Clean in a background thread (parallel)
    """

    def __init__(self, config: dict):
        self.config = config

        # Directories
        self.db_path = Path(config.get('db_path') or DATA_DIR / "db/db")
        self.lemmatized_dir = Path(config.get('lemmatized_dir') or PROCESSED_DATA_DIR / "camel_lemmatized")
        self.validation_dir = Path(config.get('validation_dir') or DATA_DIR / "validation_reports")
        self.clean_output_dir = Path(config.get('clean_output_dir') or PROCESSED_DATA_DIR / "clean_lemmatized")

        # Create directories
        self.lemmatized_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.clean_output_dir.mkdir(parents=True, exist_ok=True)

        # Parameters
        self.batch_size = config.get('batch_size', 500)
        self.workers = config.get('workers', 8)
        self.dialect = config.get('dialect', 'calima-msa-r13')
        self.llm_model = config.get('llm_model', 'deepseek/deepseek-r1')
        self.api_key = config.get('api_key') or os.environ.get('OPENROUTER_API_KEY')

        # Queue for passing completed batches to validation thread
        self.batch_queue = queue.Queue()

        # Shared state (thread-safe)
        self.lock = threading.Lock()
        self.stats = {
            'batches_lemmatized': 0,
            'batches_validated': 0,
            'batches_resolved': 0,
            'files_lemmatized': 0,
            'files_cleaned': 0,
            'conflicts_found': 0,
            'errors_detected': 0,
            'polysemy_confirmed': 0
        }

        # Control flags
        self.lemmatization_done = threading.Event()
        self.stop_requested = threading.Event()

        # State tracking
        self.state_file = DATA_DIR / 'pipeline/parallel_state.json'

        # Ensure state file directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.state = {}
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {'last_batch': 0}

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_unprocessed_files(self) -> list:
        """Get list of files not yet lemmatized"""
        progress_file = self.lemmatized_dir / 'progress.json'
        processed = set()
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                processed = set(json.load(f))

        all_files = []
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
                        file_key = str(file_path.relative_to(self.db_path))
                        if file_key not in processed:
                            all_files.append(file_path)
        return all_files

    # ==================== LEMMATIZATION (Main Thread) ====================

    def run_lemmatization_batch(self, batch_num: int) -> tuple:
        """Run lemmatization for one batch. Returns (files_processed, batch_file_names)"""
        logger.info(f"[LEMMATIZE] Batch {batch_num}: Starting...")

        from camel_lemmatize_fast import FastDatabaseLemmatizer

        lemmatizer = FastDatabaseLemmatizer(
            db_path=str(self.db_path),
            output_dir=str(self.lemmatized_dir),
            dialect=self.dialect,
            num_workers=self.workers
        )

        # Get files before
        progress_file = self.lemmatized_dir / 'progress.json'
        before_files = set()
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                before_files = set(json.load(f))

        # Process batch
        lemmatizer.process_database(limit=self.batch_size)

        # Get files after
        after_files = set()
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                after_files = set(json.load(f))

        # Identify new files
        new_files = after_files - before_files
        batch_file_names = []
        for file_key in new_files:
            stem = Path(file_key).stem
            json_name = f"{stem}_lemmatized.json"
            batch_file_names.append(json_name)

        files_processed = len(new_files)

        with self.lock:
            self.stats['files_lemmatized'] += files_processed
            self.stats['batches_lemmatized'] += 1

        logger.info(f"[LEMMATIZE] Batch {batch_num}: ✓ Completed {files_processed} files")
        return files_processed, batch_file_names

    def lemmatization_thread(self, max_batches: int = None):
        """Main lemmatization loop - runs continuously"""
        batch_num = self.state['last_batch']

        while not self.stop_requested.is_set():
            batch_num += 1

            if max_batches and batch_num > max_batches:
                logger.info(f"[LEMMATIZE] Reached max batches ({max_batches})")
                break

            unprocessed = self.get_unprocessed_files()
            if not unprocessed:
                logger.info("[LEMMATIZE] ✓ All files processed!")
                break

            logger.info(f"\n[LEMMATIZE] ===== BATCH {batch_num} ({len(unprocessed)} remaining) =====")

            try:
                files_processed, batch_files = self.run_lemmatization_batch(batch_num)

                if files_processed == 0:
                    logger.info("[LEMMATIZE] No new files, stopping")
                    break

                # Put batch info in queue for validation thread
                self.batch_queue.put({
                    'batch_num': batch_num,
                    'files': batch_files,
                    'count': files_processed
                })

                self.state['last_batch'] = batch_num
                self.save_state()

            except Exception as e:
                logger.error(f"[LEMMATIZE] Error in batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                break

        # Signal that lemmatization is done
        self.lemmatization_done.set()
        # Put sentinel to stop validation thread
        self.batch_queue.put(None)
        logger.info("[LEMMATIZE] Thread finished")

    # ==================== VALIDATION + LLM + CLEAN (Background Thread) ====================

    def run_validation(self, batch_num: int, batch_files: list) -> dict:
        """Run embedding-based validation on batch files"""
        logger.info(f"[VALIDATE] Batch {batch_num}: Starting validation of {len(batch_files)} files...")

        from validate_lemmatization import LemmatizationValidator

        batch_validation_dir = self.validation_dir / f'batch_{batch_num:04d}'
        batch_validation_dir.mkdir(exist_ok=True)

        validator = LemmatizationValidator(
            lemmatized_dir=str(self.lemmatized_dir),
            output_dir=str(batch_validation_dir)
        )

        validator.load_lemmatized_files_by_names(batch_files)
        conflicts = validator.find_pos_conflicts()

        if not conflicts:
            logger.info(f"[VALIDATE] Batch {batch_num}: No POS conflicts found")
            return {'conflicts': 0, 'results': None, 'validation_dir': batch_validation_dir}

        results_df = validator.validate_conflicts(conflicts, max_samples_per_pos=3)
        validator.generate_reports(results_df)

        with self.lock:
            self.stats['conflicts_found'] += len(results_df)
            self.stats['batches_validated'] += 1

        logger.info(f"[VALIDATE] Batch {batch_num}: ✓ Found {len(conflicts)} lemmas with conflicts")
        return {'conflicts': len(results_df), 'results': results_df, 'validation_dir': batch_validation_dir}

    def run_llm_resolution(self, batch_num: int, validation_dir: Path) -> dict:
        """Run LLM resolution for uncertain cases"""
        if not self.api_key:
            logger.warning(f"[LLM] Batch {batch_num}: No API key, skipping")
            return {'resolved': 0}

        logger.info(f"[LLM] Batch {batch_num}: Starting LLM resolution...")

        from llm_resolve_lemmas import LemmaResolver, load_validation_results

        validation_csv = validation_dir / 'validation_results.csv'
        if not validation_csv.exists():
            return {'resolved': 0}

        cases = load_validation_results(str(validation_csv))
        uncertain_cases = [c for c in cases if c.status == 'UNCERTAIN']

        if not uncertain_cases:
            logger.info(f"[LLM] Batch {batch_num}: No uncertain cases")
            return {'resolved': 0}

        resolver = LemmaResolver(self.api_key, self.llm_model)
        resolver.resolve_batch(uncertain_cases, delay=0.3)

        output_dir = validation_dir / 'llm_resolved'
        resolver.save_results(str(output_dir))

        errors = sum(1 for r in resolver.results if r.get('decision') == 'ERROR')
        polysemy = sum(1 for r in resolver.results if r.get('decision') == 'POLYSEMY')

        with self.lock:
            self.stats['errors_detected'] += errors
            self.stats['polysemy_confirmed'] += polysemy
            self.stats['batches_resolved'] += 1

        logger.info(f"[LLM] Batch {batch_num}: ✓ Resolved {len(uncertain_cases)} cases (errors={errors}, polysemy={polysemy})")
        return {'resolved': len(uncertain_cases), 'errors': errors, 'polysemy': polysemy}

    def generate_clean_output(self, batch_num: int, batch_files: list) -> int:
        """Generate clean lemmatized TEXT files with original names and POS tags"""
        logger.info(f"[CLEAN] Batch {batch_num}: Generating clean output...")

        # Collect all LLM corrections
        corrections = {}
        for batch_dir in self.validation_dir.glob('batch_*'):
            resolutions_file = batch_dir / 'llm_resolved' / 'llm_resolutions.json'
            if resolutions_file.exists():
                with open(resolutions_file, 'r', encoding='utf-8') as f:
                    resolutions = json.load(f)
                for r in resolutions:
                    if r.get('decision') == 'ERROR' and r.get('correct_pos'):
                        key = (r['lemma'], r['pos1'], r['pos2'])
                        corrections[key] = r['correct_pos']

        # Process only this batch's files
        cleaned_count = 0
        for json_name in batch_files:
            json_file = self.lemmatized_dir / json_name
            if not json_file.exists():
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get original filename (without _lemmatized suffix)
                source_file = data.get('source_file', '')
                if source_file:
                    original_name = Path(source_file).name
                else:
                    original_name = json_name.replace('_lemmatized.json', '')

                # Output as text file with original name
                output_file = self.clean_output_dir / original_name

                if output_file.exists():
                    continue

                # Apply corrections and build lemmatized text with POS
                words_and_lemmas = data.get('words_and_lemmas', [])
                lemmatized_tokens = []

                for item in words_and_lemmas:
                    lemma = item.get('lemma', '')
                    pos = item.get('pos', 'UNKNOWN')

                    # Apply corrections if any
                    for (corr_lemma, pos1, pos2), correct_pos in corrections.items():
                        if lemma == corr_lemma and pos in [pos1, pos2]:
                            pos = correct_pos
                            break

                    # Use lemma for output (or original word if no lemma)
                    token = lemma if lemma else item.get('word', '')

                    # Format: lemma/POS
                    lemmatized_tokens.append(f"{token}/{pos}")

                # Write as plain text (space-separated lemma/POS pairs)
                lemmatized_text = ' '.join(lemmatized_tokens)

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(lemmatized_text)

                cleaned_count += 1

            except Exception as e:
                logger.error(f"[CLEAN] Error cleaning {json_name}: {e}")

        with self.lock:
            self.stats['files_cleaned'] += cleaned_count

        logger.info(f"[CLEAN] Batch {batch_num}: ✓ Generated {cleaned_count} clean text files (lemma/POS format)")
        return cleaned_count

    def validation_thread(self):
        """Background thread for validation + LLM + clean"""
        logger.info("[VALIDATE] Thread started - waiting for batches...")

        while True:
            try:
                # Wait for a batch from the queue (blocks until available)
                batch_info = self.batch_queue.get(timeout=1)

                if batch_info is None:
                    # Sentinel received - lemmatization is done
                    logger.info("[VALIDATE] Received stop signal")
                    break

                batch_num = batch_info['batch_num']
                batch_files = batch_info['files']

                logger.info(f"\n[VALIDATE] ===== Processing Batch {batch_num} =====")

                # Step 1: Validation
                validation_result = self.run_validation(batch_num, batch_files)

                # Step 2: LLM Resolution (if conflicts)
                if validation_result['conflicts'] > 0:
                    self.run_llm_resolution(batch_num, validation_result['validation_dir'])

                # Step 3: Clean output
                self.generate_clean_output(batch_num, batch_files)

                logger.info(f"[VALIDATE] Batch {batch_num}: ✓ Complete")

            except queue.Empty:
                # Check if lemmatization is done and queue is empty
                if self.lemmatization_done.is_set() and self.batch_queue.empty():
                    break
                continue
            except Exception as e:
                logger.error(f"[VALIDATE] Error: {e}")
                import traceback
                traceback.print_exc()

        logger.info("[VALIDATE] Thread finished")

    def print_summary(self):
        """Print final summary"""
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE SUMMARY")
        logger.info(f"{'='*80}")

        with self.lock:
            logger.info(f"Batches lemmatized:      {self.stats['batches_lemmatized']}")
            logger.info(f"Batches validated:       {self.stats['batches_validated']}")
            logger.info(f"Batches LLM resolved:    {self.stats['batches_resolved']}")
            logger.info(f"Files lemmatized:        {self.stats['files_lemmatized']:,}")
            logger.info(f"Files cleaned:           {self.stats['files_cleaned']:,}")
            logger.info(f"POS conflicts found:     {self.stats['conflicts_found']:,}")
            logger.info(f"Errors detected by LLM:  {self.stats['errors_detected']:,}")
            logger.info(f"Polysemy confirmed:      {self.stats['polysemy_confirmed']:,}")

        logger.info(f"\nOutput directories:")
        logger.info(f"  Lemmatized:  {self.lemmatized_dir}")
        logger.info(f"  Validation:  {self.validation_dir}")
        logger.info(f"  Clean:       {self.clean_output_dir}")
        logger.info(f"{'='*80}")

    def run(self, max_batches: int = None):
        """Run the parallel pipeline"""
        logger.info(f"\n{'='*80}")
        logger.info("PARALLEL LEMMATIZATION PIPELINE")
        logger.info(f"{'='*80}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Workers: {self.workers}")
        logger.info(f"Dialect: {self.dialect}")
        logger.info(f"LLM Model: {self.llm_model}")
        logger.info(f"API Key: {'✓ Set' if self.api_key else '✗ Not set'}")
        logger.info(f"\nArchitecture:")
        logger.info(f"  Main thread:       Lemmatization (continuous)")
        logger.info(f"  Background thread: Validation + LLM + Clean (parallel)")
        logger.info(f"{'='*80}\n")

        # Start validation thread
        validation_worker = threading.Thread(
            target=self.validation_thread,
            name="ValidationThread",
            daemon=True
        )
        validation_worker.start()

        try:
            # Run lemmatization in main thread
            self.lemmatization_thread(max_batches)

            # Wait for validation thread to finish
            logger.info("\n[MAIN] Waiting for validation thread to finish...")
            validation_worker.join(timeout=300)  # 5 min timeout

        except KeyboardInterrupt:
            logger.info("\n[MAIN] Interrupted by user")
            self.stop_requested.set()
            self.batch_queue.put(None)

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description='Parallel Lemmatization Pipeline')
    parser.add_argument('--batch-size', type=int, default=500, help='Files per batch')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers for lemmatization')
    parser.add_argument('--dialect', default='calima-msa-r13', help='CAMeL dialect')
    parser.add_argument('--model', default='deepseek/deepseek-r1', help='LLM model')
    parser.add_argument('--max-batches', type=int, help='Max batches to process')
    parser.add_argument('--db', default=None, help='Database path')
    parser.add_argument('--output', default=None, help='Lemmatized output dir')
    parser.add_argument('--clean-output', default=None, help='Clean output dir')

    args = parser.parse_args()

    config = {
        'batch_size': args.batch_size,
        'workers': args.workers,
        'dialect': args.dialect,
        'llm_model': args.model,
        'db_path': args.db,
        'lemmatized_dir': args.output,
        'clean_output_dir': args.clean_output
    }

    pipeline = ParallelLemmatizationPipeline(config)
    pipeline.run(max_batches=args.max_batches)


if __name__ == "__main__":
    main()

