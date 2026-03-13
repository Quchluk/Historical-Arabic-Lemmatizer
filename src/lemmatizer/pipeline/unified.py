#!/usr/bin/env python3
"""
Unified Lemmatization Pipeline
===============================
Orchestrates the complete workflow:
1. CAMeL Lemmatization (batch of 500 texts)
2. Embedding-based Validation
3. LLM Resolution of uncertain cases
4. Generate clean lemmatized files

Runs continuously until all texts are processed.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'pipeline.log'),
        FlushHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


class LemmatizationPipeline:
    """
    Unified pipeline that orchestrates:
    1. Lemmatization
    2. Validation
    3. LLM Resolution
    4. Clean output generation
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
        self.llm_model = config.get('llm_model', 'google/gemini-2.0-flash-001')
        self.api_key = config.get('api_key') or os.environ.get('OPENROUTER_API_KEY')

        # State tracking
        self.state_file = DATA_DIR / 'pipeline/unified_state.json'

        # Ensure state file directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.state = {}
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'last_batch': 0,
                'processed_files': [],
                'validated_batches': [],
                'resolved_batches': []
            }

        # Statistics
        self.stats = {
            'total_batches': 0,
            'files_lemmatized': 0,
            'conflicts_found': 0,
            'errors_detected': 0,
            'polysemy_confirmed': 0,
            'files_cleaned': 0
        }


    def save_state(self):
        """Save pipeline state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_unprocessed_files(self) -> list:
        """Get list of files not yet lemmatized"""
        # Load progress from lemmatizer
        progress_file = self.lemmatized_dir / 'progress.json'
        processed = set()
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                processed = set(json.load(f))

        # Find all text files
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

    def run_lemmatization_batch(self, batch_num: int) -> tuple:
        """Run lemmatization for one batch. Returns (files_processed, batch_file_names)"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 1: LEMMATIZATION (Batch {batch_num})")
        logger.info(f"{'='*80}")

        # Import and run lemmatizer directly (faster than subprocess)
        from lemmatizer.processing.camel_fast import FastDatabaseLemmatizer

        lemmatizer = FastDatabaseLemmatizer(
            db_path=str(self.db_path),
            output_dir=str(self.lemmatized_dir),
            dialect=self.dialect,
            num_workers=self.workers
        )

        # Get files before processing
        progress_file = self.lemmatized_dir / 'progress.json'
        before_files = set()
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                before_files = set(json.load(f))

        # Process batch
        lemmatizer.process_database(limit=self.batch_size)

        # Get files after processing
        after_files = set()
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                after_files = set(json.load(f))

        # Identify new files from this batch
        new_files = after_files - before_files
        batch_file_names = []
        for file_key in new_files:
            # Convert file key to lemmatized JSON filename
            stem = Path(file_key).stem
            json_name = f"{stem}_lemmatized.json"
            batch_file_names.append(json_name)

        files_processed = len(new_files)
        self.stats['files_lemmatized'] += files_processed

        logger.info(f"✓ Lemmatized {files_processed} files in this batch")
        return files_processed, batch_file_names

    def run_validation(self, batch_num: int, batch_files: list) -> dict:
        """Run embedding-based validation on specific batch files"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 2: VALIDATION (Batch {batch_num})")
        logger.info(f"{'='*80}")

        from lemmatizer.validation.lemmatization import LemmatizationValidator

        # Create batch-specific output directory
        batch_validation_dir = self.validation_dir / f'batch_{batch_num:04d}'
        batch_validation_dir.mkdir(exist_ok=True)

        validator = LemmatizationValidator(
            lemmatized_dir=str(self.lemmatized_dir),
            output_dir=str(batch_validation_dir)
        )

        # Load only the files from this batch
        logger.info(f"Loading {len(batch_files)} files from batch {batch_num}...")
        validator.load_lemmatized_files_by_names(batch_files)

        # Find conflicts
        conflicts = validator.find_pos_conflicts()

        if not conflicts:
            logger.info("✓ No POS conflicts found")
            return {'conflicts': 0, 'results': None, 'validation_dir': batch_validation_dir}

        # Validate with embeddings (limit to top 100 conflicts)
        results_df = validator.validate_conflicts(conflicts, max_samples_per_pos=3)

        # Save results
        validator.generate_reports(results_df)

        self.stats['conflicts_found'] += len(results_df)

        logger.info(f"✓ Found {len(conflicts)} lemmas with POS conflicts")
        logger.info(f"✓ Validated {len(results_df)} conflict pairs")

        return {'conflicts': len(results_df), 'results': results_df, 'validation_dir': batch_validation_dir}

    def run_llm_resolution(self, batch_num: int, validation_dir: Path) -> dict:
        """Run LLM resolution for uncertain cases from a specific batch"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 3: LLM RESOLUTION (Batch {batch_num})")
        logger.info(f"{'='*80}")

        if not self.api_key:
            logger.warning("⚠ No API key - skipping LLM resolution")
            return {'resolved': 0}

        from lemmatizer.processing.llm_resolve import LemmaResolver, load_validation_results

        # Load validation results from batch-specific directory
        validation_csv = validation_dir / 'validation_results.csv'
        if not validation_csv.exists():
            logger.info(f"No validation results found at {validation_csv}")
            return {'resolved': 0}

        cases = load_validation_results(str(validation_csv))

        # Filter to uncertain cases only
        uncertain_cases = [c for c in cases if c.status == 'UNCERTAIN']

        if not uncertain_cases:
            logger.info("✓ No uncertain cases to resolve")
            return {'resolved': 0}

        logger.info(f"Resolving {len(uncertain_cases)} uncertain cases with LLM...")

        # Create resolver
        resolver = LemmaResolver(self.api_key, self.llm_model)

        # Resolve (with rate limiting)
        resolver.resolve_batch(uncertain_cases, delay=0.3)

        # Save results to batch-specific directory
        output_dir = validation_dir / 'llm_resolved'
        resolver.save_results(str(output_dir))

        # Count results
        errors = sum(1 for r in resolver.results if r.get('decision') == 'ERROR')
        polysemy = sum(1 for r in resolver.results if r.get('decision') == 'POLYSEMY')

        self.stats['errors_detected'] += errors
        self.stats['polysemy_confirmed'] += polysemy

        logger.info(f"✓ Resolved {len(uncertain_cases)} cases")
        logger.info(f"  - Errors detected: {errors}")
        logger.info(f"  - Polysemy confirmed: {polysemy}")

        return {'resolved': len(uncertain_cases), 'errors': errors, 'polysemy': polysemy}

    def generate_clean_output(self, batch_num: int) -> int:
        """Generate clean lemmatized files with ORIGINAL file names from DB"""
        logger.info(f"\n{'='*80}")
        logger.info(f"STEP 4: GENERATE CLEAN OUTPUT (Batch {batch_num})")
        logger.info(f"{'='*80}")

        # Collect LLM corrections from ALL batch directories
        corrections = {}

        for batch_dir in self.validation_dir.glob('batch_*'):
            resolutions_file = batch_dir / 'llm_resolved' / 'llm_resolutions.json'
            if resolutions_file.exists():
                with open(resolutions_file, 'r', encoding='utf-8') as f:
                    resolutions = json.load(f)

                # Build correction map for errors
                for r in resolutions:
                    if r.get('decision') == 'ERROR' and r.get('correct_pos'):
                        key = (r['lemma'], r['pos1'], r['pos2'])
                        corrections[key] = r['correct_pos']

        logger.info(f"Loaded {len(corrections)} corrections from LLM (all batches)")

        # Process lemmatized files
        cleaned_count = 0
        lemmatized_files = list(self.lemmatized_dir.glob('*_lemmatized.json'))

        for json_file in lemmatized_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get ORIGINAL file name from source_file path
                source_file = data.get('source_file', '')
                if source_file:
                    # Extract the original file name (e.g., "0001Quran.Mushaf.Tanzil001-ara1")
                    original_name = Path(source_file).name
                else:
                    # Fallback: remove "_lemmatized" suffix
                    original_name = json_file.stem.replace('_lemmatized', '')

                # Create output path with original name + .json extension
                output_file = self.clean_output_dir / f"{original_name}.json"

                # Skip if already cleaned
                if output_file.exists():
                    continue

                # Apply corrections if any
                words_and_lemmas = data.get('words_and_lemmas', [])
                corrections_applied = 0

                for item in words_and_lemmas:
                    lemma = item.get('lemma', '')
                    pos = item.get('pos', '')

                    # Check if this needs correction
                    for (corr_lemma, pos1, pos2), correct_pos in corrections.items():
                        if lemma == corr_lemma and pos in [pos1, pos2]:
                            item['original_pos'] = pos
                            item['pos'] = correct_pos
                            item['corrected'] = True
                            corrections_applied += 1
                            break

                # Add metadata
                data['original_file'] = source_file
                data['cleaned_at'] = datetime.now().isoformat()
                data['corrections_applied'] = corrections_applied

                # Save clean version with ORIGINAL name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)

                cleaned_count += 1

            except Exception as e:
                logger.error(f"Error cleaning {json_file.name}: {e}")

        self.stats['files_cleaned'] += cleaned_count
        logger.info(f"✓ Generated {cleaned_count} clean lemmatized files (original names)")

        return cleaned_count

    def print_summary(self):
        """Print pipeline summary"""
        logger.info(f"\n{'='*80}")
        logger.info("PIPELINE SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total batches processed: {self.stats['total_batches']}")
        logger.info(f"Files lemmatized:        {self.stats['files_lemmatized']:,}")
        logger.info(f"POS conflicts found:     {self.stats['conflicts_found']:,}")
        logger.info(f"Errors detected by LLM:  {self.stats['errors_detected']:,}")
        logger.info(f"Polysemy confirmed:      {self.stats['polysemy_confirmed']:,}")
        logger.info(f"Clean files generated:   {self.stats['files_cleaned']:,}")
        logger.info(f"\nOutput directories:")
        logger.info(f"  Lemmatized:  {self.lemmatized_dir}")
        logger.info(f"  Validation:  {self.validation_dir}")
        logger.info(f"  Clean:       {self.clean_output_dir}")
        logger.info(f"{'='*80}")

    def run(self, max_batches: int = None):
        """Run the complete pipeline"""
        logger.info(f"\n{'='*80}")
        logger.info("UNIFIED LEMMATIZATION PIPELINE")
        logger.info(f"{'='*80}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Workers: {self.workers}")
        logger.info(f"Dialect: {self.dialect}")
        logger.info(f"LLM Model: {self.llm_model}")
        logger.info(f"API Key: {'✓ Set' if self.api_key else '✗ Not set'}")

        batch_num = self.state['last_batch']

        while True:
            batch_num += 1

            if max_batches and batch_num > max_batches:
                logger.info(f"\nReached max batches ({max_batches})")
                break

            # Check if there are files to process
            unprocessed = self.get_unprocessed_files()
            if not unprocessed:
                logger.info("\n✓ All files have been processed!")
                break

            logger.info(f"\n{'#'*80}")
            logger.info(f"BATCH {batch_num} - {len(unprocessed)} files remaining")
            logger.info(f"{'#'*80}")

            try:
                # Step 1: Lemmatization
                files_processed, batch_files = self.run_lemmatization_batch(batch_num)

                if files_processed == 0:
                    logger.info("No new files processed, stopping")
                    break

                # Step 2: Validation (on this batch's files)
                validation_result = self.run_validation(batch_num, batch_files)

                # Step 3: LLM Resolution (if conflicts found)
                if validation_result['conflicts'] > 0:
                    self.run_llm_resolution(batch_num, validation_result['validation_dir'])

                # Step 4: Generate clean output
                self.generate_clean_output(batch_num)

                # Update state
                self.state['last_batch'] = batch_num
                self.save_state()
                self.stats['total_batches'] += 1

                # Print batch summary
                logger.info(f"\n✓ Batch {batch_num} complete!")

            except KeyboardInterrupt:
                logger.info("\n⚠ Pipeline interrupted by user")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"\n✗ Error in batch {batch_num}: {e}")
                import traceback
                traceback.print_exc()
                self.save_state()
                break

        # Final summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description='Unified Lemmatization Pipeline')
    parser.add_argument('--batch-size', type=int, default=500, help='Files per batch')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
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

    pipeline = LemmatizationPipeline(config)
    pipeline.run(max_batches=args.max_batches)


if __name__ == "__main__":
    main()

