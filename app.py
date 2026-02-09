#!/usr/bin/env python3
"""
On-the-Fly Word Context Analyzer
Generates embeddings, analyzes, and discards them immediately
No need to store embeddings on disk
"""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import logging
import gc
import gzip
import re

# Import from embed_texts.py
from embed_texts import TextExtractor, AraBERTEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('word_context_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OnTheFlyAnalyzer:
    """Analyze word context diversity without storing embeddings on disk"""

    def __init__(self, metadata_file: str, output_dir: str = 'word_context_analysis'):
        self.metadata_file = metadata_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load stop words and non-Arabic words
        logger.info("Loading filter lists...")
        self.arabic_stop_words = self._load_stop_words('db_training/stop_words/arabic_stop_words.txt')
        self.non_arabic_words = self._load_non_arabic_words('db_training/stop_words/non_arabic_words_list.txt')
        logger.info(f"Loaded {len(self.arabic_stop_words)} Arabic stop words")
        logger.info(f"Loaded {len(self.non_arabic_words)} non-Arabic word patterns")

        # Batch accumulator (for ~10-15 texts at a time)
        self.current_batch = defaultdict(lambda: {
            'occurrences': [],
            'texts': []
        })
        self.batch_num = 0

        # Output files
        self.text_stats_file = self.output_dir / 'per_text_statistics.csv'
        self.word_batch_dir = self.output_dir / 'word_batches'
        self.word_batch_dir.mkdir(exist_ok=True)

        # Initialize CSV files with headers
        self._initialize_output_files()

        # Initialize embedder (reuse for all texts)
        logger.info("Initializing AraBERT embedder...")
        self.embedder = AraBERTEmbedder()
        self.text_extractor = TextExtractor()

        # Processing stats
        self.stats = {
            'total_texts': 0,
            'processed': 0,
            'errors': 0,
            'total_words': 0,
            'filtered_words': 0
        }

    def _load_stop_words(self, filepath: str) -> set:
        """Load stop words from a text file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                words = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(words)} words from {filepath}")
            return words
        except FileNotFoundError:
            logger.warning(f"Stop words file not found: {filepath}")
            return set()

    def _load_non_arabic_words(self, filepath: str) -> set:
        """Load non-Arabic words list and extract Arabic-only patterns."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_words = [line.strip() for line in f if line.strip()]

            # Extract only Arabic characters from each word
            arabic_patterns = set()
            for word in raw_words:
                arabic_only = self._extract_arabic_only(word)
                if arabic_only:
                    arabic_patterns.add(arabic_only)

            logger.info(f"Extracted {len(arabic_patterns)} Arabic patterns from {filepath}")
            return arabic_patterns
        except FileNotFoundError:
            logger.warning(f"Non-Arabic words file not found: {filepath}")
            return set()

    @staticmethod
    def _extract_arabic_only(text: str) -> str:
        """Extract only Arabic characters from text, removing numbers and other characters."""
        arabic_chars = re.findall(r'[\u0600-\u06FF]+', text)
        return ''.join(arabic_chars)

    def _should_filter_word(self, word: str) -> bool:
        """Determine if a word should be filtered out."""
        # Filter if it's an Arabic stop word
        if word in self.arabic_stop_words:
            return True

        # Extract Arabic-only version and check against non-Arabic words list
        arabic_only = self._extract_arabic_only(word)
        if arabic_only and arabic_only in self.non_arabic_words:
            return True

        return False


    def _initialize_output_files(self):
        """Initialize output CSV files with headers"""
        # Text statistics file
        with open(self.text_stats_file, 'w', encoding='utf-8') as f:
            f.write('text_id,total_words,unique_words\n')

        logger.info(f"Initialized output files in {self.output_dir}")

    def process_single_text(self, file_path: str, version_uri: str):
        """Process one text: extract, embed, and accumulate in current batch"""
        try:
            # 1. Extract text content (excluding metadata)
            text_content, _ = self.text_extractor.extract_text_content(file_path)

            if not text_content or len(text_content.strip()) < 10:
                logger.warning(f"Skipping {version_uri}: insufficient content")
                return False

            # 2. Generate word-level embeddings
            word_embeddings, words, pooled_embedding = self.embedder.embed_text(text_content)

            if len(words) == 0:
                logger.warning(f"Skipping {version_uri}: no words extracted")
                return False

            # 3. Accumulate word embeddings in current batch (compressed to float16)
            # Apply filtering before accumulation
            text_unique_words = set()
            filtered_count = 0

            for word, embedding in zip(words, word_embeddings):
                # Skip if word should be filtered
                if self._should_filter_word(word):
                    filtered_count += 1
                    continue

                self.current_batch[word]['occurrences'].append(embedding.astype(np.float16))
                self.current_batch[word]['texts'].append(version_uri)
                text_unique_words.add(word)

            # Update filtering stats
            self.stats['filtered_words'] += filtered_count

            if filtered_count > 0:
                logger.debug(f"Filtered {filtered_count} words from {version_uri}")

            # 4. Write text statistics
            with open(self.text_stats_file, 'a', encoding='utf-8') as f:
                f.write(f'{version_uri},{len(words)},{len(text_unique_words)}\n')

            self.stats['total_words'] += len(words)

            # 5. Free embeddings memory
            del word_embeddings, words, pooled_embedding, text_content
            gc.collect()

            return True

        except Exception as e:
            logger.error(f"Error processing {version_uri}: {e}")
            return False

    def save_current_batch(self):
        """Save current batch: compute mean pooling per word per text, then clear memory"""
        if not self.current_batch:
            return

        logger.info(f"Saving batch {self.batch_num} ({len(self.current_batch):,} unique words)...")

        # Group by (word, text) to compute mean pooling
        batch_file = self.word_batch_dir / f'batch_{self.batch_num:04d}.csv.gz'

        with gzip.open(batch_file, 'wt', encoding='utf-8') as f:
            f.write('word,text_id,num_occurrences,avg_embedding\n')

            for word, data in self.current_batch.items():
                # Group occurrences by text
                text_occurrences = defaultdict(list)
                for emb, text_id in zip(data['occurrences'], data['texts']):
                    text_occurrences[text_id].append(emb)

                # Write one row per (word, text) with mean pooling
                for text_id, embeddings in text_occurrences.items():
                    avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
                    emb_str = ','.join(map(str, avg_emb))
                    f.write(f'{word},{text_id},{len(embeddings)},"{emb_str}"\n')

        logger.info(f"Saved batch {self.batch_num}")

        # Clear batch and free memory
        self.current_batch.clear()
        gc.collect()
        self.batch_num += 1

    def analyze_word_from_aggregates(self, word: str, data: dict):
        """Analyze word from aggregated data across texts"""
        embeddings = np.array(data['embeddings'])

        if len(embeddings) < 2:
            return {
                'word': word,
                'total_occurrences': data['total_occurrences'],
                'num_contexts': 1,
                'num_texts': data['num_texts'],
                'avg_similarity': 1.0,
                'context_diversity_ratio': 0.0,
                'context_diversity': 'single_occurrence'
            }

        # Calculate similarity
        similarities = cosine_similarity(embeddings)
        mask = ~np.eye(len(embeddings), dtype=bool)
        avg_sim = similarities[mask].mean()

        # Cluster
        clustering = DBSCAN(eps=0.15, min_samples=2, metric='cosine').fit(embeddings)
        num_contexts = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        if num_contexts == 0:
            num_contexts = len(embeddings)

        ratio = num_contexts / data['total_occurrences']

        diversity = ('monosemous' if ratio < 0.1 else
                    'low_variation' if ratio < 0.3 else
                    'moderate_variation' if ratio < 0.5 else
                    'high_variation' if ratio < 0.7 else 'polysemous')

        del embeddings, similarities

        return {
            'word': word,
            'total_occurrences': data['total_occurrences'],
            'num_contexts': num_contexts,
            'num_texts': data['num_texts'],
            'avg_similarity': avg_sim,
            'context_diversity_ratio': ratio,
            'context_diversity': diversity
        }

    def process_all_texts(self, texts_per_batch=10):
        """Process all texts in batches optimized for 5-6GB RAM"""
        # Load metadata
        logger.info(f"Loading metadata from: {self.metadata_file}")
        metadata_df = pd.read_parquet(self.metadata_file)

        # Filter valid texts
        valid_texts = metadata_df[metadata_df['text_file_path'] != '']
        self.stats['total_texts'] = len(valid_texts)

        logger.info(f"Processing {self.stats['total_texts']:,} texts...")
        logger.info(f"Batch size: {texts_per_batch} texts per batch (optimized for 5-6GB RAM)")
        logger.info("="*80)

        # Process texts in batches
        for idx, row in tqdm(valid_texts.iterrows(), total=len(valid_texts), desc="Processing"):
            success = self.process_single_text(row['text_file_path'], row['version_uri'])

            if success:
                self.stats['processed'] += 1
            else:
                self.stats['errors'] += 1

            # Save batch when we reach batch size
            if self.stats['processed'] % texts_per_batch == 0:
                logger.info(f"\nProcessed {self.stats['processed']:,}/{self.stats['total_texts']:,} texts")
                self.save_current_batch()

            # Force GC every 5 texts
            if self.stats['processed'] % 5 == 0:
                gc.collect()

        # Save final batch if any remaining
        if self.current_batch:
            logger.info(f"\nSaving final batch...")
            self.save_current_batch()

        logger.info("\nAll texts processed. Aggregating results...")
        self.aggregate_all_batches()


    def aggregate_all_batches(self):
        """Aggregate word statistics across all batch files in streaming mode"""
        batch_files = sorted(self.word_batch_dir.glob('batch_*.csv.gz'))

        logger.info(f"Aggregating {len(batch_files)} batch files in streaming mode...")

        # Step 1: Collect word occurrence counts first (lightweight - just metadata)
        logger.info("Pass 1: Collecting word occurrence counts...")
        word_occurrences = defaultdict(lambda: {'total_occurrences': 0, 'num_texts': 0, 'file_locations': []})

        for bf in tqdm(batch_files, desc="Scanning batches"):
            df = pd.read_csv(bf, compression='gzip')

            for _, row in df.iterrows():
                word = row['word']
                word_occurrences[word]['total_occurrences'] += row['num_occurrences']
                word_occurrences[word]['num_texts'] += 1
                word_occurrences[word]['file_locations'].append((str(bf), word))

        logger.info(f"Found {len(word_occurrences):,} unique words")

        # Step 2: Process words in batches to avoid RAM overflow
        logger.info("Pass 2: Analyzing words in batches...")
        results = []
        words_to_process = list(word_occurrences.keys())
        word_batch_size = 3000  # Process 3000 words at a time (conservative for 5-6GB RAM)

        for batch_start in tqdm(range(0, len(words_to_process), word_batch_size), desc="Word batches"):
            batch_words = words_to_process[batch_start:batch_start + word_batch_size]
            word_embeddings = defaultdict(list)

            # Determine which files we need to read for this word batch
            files_to_read = set()
            for word in batch_words:
                for file_path, _ in word_occurrences[word]['file_locations']:
                    files_to_read.add(file_path)

            # Read only the required files
            for file_path in files_to_read:
                df = pd.read_csv(file_path, compression='gzip')

                for _, row in df.iterrows():
                    if row['word'] in batch_words:
                        emb = np.fromstring(row['avg_embedding'], sep=',', dtype=np.float32)
                        word_embeddings[row['word']].append(emb)

            # Analyze this batch of words
            for word in batch_words:
                data = {
                    'total_occurrences': word_occurrences[word]['total_occurrences'],
                    'num_texts': word_occurrences[word]['num_texts'],
                    'embeddings': word_embeddings[word]
                }
                result = self.analyze_word_from_aggregates(word, data)
                results.append(result)

            # Free memory after each word batch
            del word_embeddings
            gc.collect()

        final_df = pd.DataFrame(results)
        self.save_final_results(final_df)

        # Clean up batch files
        logger.info("Cleaning up temporary batch files...")
        for bf in batch_files:
            bf.unlink()
        self.word_batch_dir.rmdir()
        logger.info("✓ Cleanup complete")

    def save_final_results(self, df):
        """Save final analysis results"""
        # Sort by occurrences
        df = df.sort_values('total_occurrences', ascending=False)

        # Save word analysis
        word_file = self.output_dir / 'word_context_diversity_full.csv'
        df.to_csv(word_file, index=False, encoding='utf-8-sig')
        logger.info(f"\nSaved word analysis: {len(df):,} words to {word_file}")

        # Summary
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Total texts processed: {self.stats['processed']:,}")
        logger.info(f"Total errors: {self.stats['errors']}")
        logger.info(f"Total unique words analyzed: {len(df):,}")
        logger.info(f"Total words filtered (stop words + non-Arabic): {self.stats['filtered_words']:,}")
        logger.info(f"Total word occurrences: {df['total_occurrences'].sum():,}")
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info(f"  - {word_file.name}")
        logger.info(f"  - {self.text_stats_file.name}")

        logger.info("\nDiversity Distribution:")
        for cat, count in df['context_diversity'].value_counts().items():
            pct = (count / len(df)) * 100
            logger.info(f"  {cat:20s}: {count:6,} ({pct:5.2f}%)")

        logger.info("\nTop 30 Most Frequent Words:")
        print(df[['word', 'total_occurrences', 'num_contexts', 'num_texts', 'context_diversity']].head(30).to_string(index=False))

        logger.info("\nTop 30 Most Polysemous Words (min 50 occurrences):")
        poly = df[df['total_occurrences'] >= 50].sort_values('num_contexts', ascending=False)
        if len(poly) > 0:
            print(poly[['word', 'total_occurrences', 'num_contexts', 'context_diversity_ratio', 'context_diversity']].head(30).to_string(index=False))


def main():
    logger.info("="*80)
    logger.info("OPTIMIZED WORD CONTEXT ANALYZER")
    logger.info("Processing one text at a time - minimal RAM usage")
    logger.info("="*80)

    analyzer = OnTheFlyAnalyzer(
        metadata_file="openiti_metadata.parquet",
        output_dir="jobs/TF_stat/word_context_analysis"
    )

    # Process all texts (one at a time, no batching)
    analyzer.process_all_texts(texts_per_batch=1)

    logger.info("\n✓ Complete!")


if __name__ == "__main__":
    main()

