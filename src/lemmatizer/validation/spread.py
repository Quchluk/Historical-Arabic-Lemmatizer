#!/usr/bin/env python3
"""
Lemma Spread Validator using AraBERT Embeddings
================================================

Validates lemmatization quality by analyzing the embedding space:

1. INTRA-LEMMA SPREAD: How far apart are word forms of the SAME lemma?
   - Good lemmatization → forms cluster tightly together
   - Bad lemmatization → forms are scattered (wrong words grouped together)

2. INTER-LEMMA DISTANCE: How far is this lemma from its NEIGHBORS?
   - Good lemmatization → lemmas are well-separated
   - Bad lemmatization → different lemmas overlap

3. SPREAD RATIO: intra_spread / inter_distance
   - Low ratio (<0.3) → tight, well-separated clusters = GOOD
   - High ratio (>0.7) → loose, overlapping clusters = SUSPICIOUS
   - Ratio > 1.0 → forms more spread than distance to neighbors = LIKELY ERROR

Follows Zipf's Law:
- Frequent lemmas: More forms, should still cluster
- Rare lemmas: Few forms, harder to validate but should be distant from others

Memory Management:
- Processes lemmas in batches
- Generates embeddings, computes metrics, then deletes embeddings
- Saves only final statistics, not embeddings
"""

import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import logging
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import gc
import math
import argparse
from datetime import datetime
from lemmatizer.config import LOGS_DIR, DATA_DIR, PROCESSED_DATA_DIR, DB_TRAINING_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'validate_lemma_spread.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LemmaSpreadStats:
    """Statistics for a single lemma's embedding spread"""
    lemma: str
    total_forms: int
    total_occurrences: int

    # Intra-lemma metrics (spread within the lemma)
    intra_mean_distance: float = 0.0  # Mean pairwise distance between forms
    intra_max_distance: float = 0.0   # Max distance (diameter)
    intra_std_distance: float = 0.0   # Std dev of distances
    centroid_spread: float = 0.0      # Mean distance from centroid

    # Inter-lemma metrics (distance to neighbors)
    nearest_neighbor_distance: float = 0.0  # Distance to closest other lemma
    nearest_neighbor_lemma: str = ""
    mean_neighbor_distance: float = 0.0     # Mean distance to k nearest lemmas

    # Quality metrics
    spread_ratio: float = 0.0         # intra_spread / nearest_neighbor_distance
    silhouette_score: float = 0.0     # Cluster quality (-1 to 1)
    quality_status: str = "UNKNOWN"   # GOOD, SUSPICIOUS, LIKELY_ERROR

    # Sample data
    sample_forms: List[str] = field(default_factory=list)
    pos_distribution: Dict[str, int] = field(default_factory=dict)
    outlier_forms: List[str] = field(default_factory=list)  # Forms far from centroid


class EmbeddingGenerator:
    """Generates embeddings using AraBERT, with memory management"""

    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2"):
        logger.info(f"Initializing AraBERT model: {model_name}")

        # Auto-detect device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        logger.info(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts, return numpy array"""
        if not texts:
            return np.array([])

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                max_length=128,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            all_embeddings.append(embeddings.cpu().numpy())

            # Clear GPU cache periodically
            if i % (batch_size * 10) == 0 and i > 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return np.vstack(all_embeddings)

    def cleanup(self):
        """Free GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class LemmaSpreadValidator:
    """Main validator for lemma spread analysis"""

    def __init__(self, lemmatized_dir: str = None,
                 output_dir: str = None,
                 stop_words_dir: str = None):
        self.lemmatized_dir = Path(lemmatized_dir) if lemmatized_dir else PROCESSED_DATA_DIR / "camel_lemmatized"
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR / "spread_validation"

        if stop_words_dir == "DISABLED":
            self.stop_words_dir = None
        else:
            self.stop_words_dir = Path(stop_words_dir) if stop_words_dir else DB_TRAINING_DIR / "stop_words"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load stop words
        self.stop_words = self._load_stop_words() if self.stop_words_dir else set()

        # Storage: lemma -> list of (form, context, pos)
        self.lemma_data: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

        # Lemma centroids (computed during processing)
        self.lemma_centroids: Dict[str, np.ndarray] = {}

        # Results
        self.lemma_stats: Dict[str, LemmaSpreadStats] = {}

        # Global stats
        self.global_stats = {
            'total_files': 0,
            'total_words': 0,
            'unique_lemmas': 0,
            'unique_forms': 0,
            'stop_words_loaded': len(self.stop_words),
            'lemmas_filtered_stopwords': 0,
            'good_lemmas': 0,
            'suspicious_lemmas': 0,
            'likely_errors': 0,
            'processing_time': 0
        }

        # Thresholds (calibrated for Arabic + AraBERT contextual embeddings)
        # Note: Contextual embeddings create wider spread than static embeddings
        # because the same word in different contexts gets different vectors.
        # Spread ratio > 1 is normal for function words (في, من, على)
        self.SPREAD_RATIO_GOOD = 3.0        # Below this = good clustering
        self.SPREAD_RATIO_SUSPICIOUS = 6.0   # Above this = suspicious
        self.SPREAD_RATIO_ERROR = 10.0       # Above this = likely error

        # Minimum samples
        self.MIN_FORMS_FOR_SPREAD = 3   # Need at least 3 forms to compute spread
        self.MIN_FORMS_FOR_OUTLIER = 5  # Need 5+ forms to detect outliers

        # Embedder (lazy load)
        self._embedder = None

    def _load_stop_words(self) -> set:
        """Load stop words from Arabic and non-Arabic lists"""
        stop_words = set()

        # Arabic stop words
        arabic_file = self.stop_words_dir / 'arabic_stop_words.txt'
        if arabic_file.exists():
            try:
                with open(arabic_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word and not word.startswith('#'):
                            stop_words.add(word)
                logger.info(f"Loaded {len(stop_words)} Arabic stop words")
            except Exception as e:
                logger.warning(f"Error loading Arabic stop words: {e}")

        # Non-Arabic words
        non_arabic_file = self.stop_words_dir / 'non_arabic_words_list.txt'
        if non_arabic_file.exists():
            try:
                count_before = len(stop_words)
                with open(non_arabic_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word and not word.startswith('#'):
                            stop_words.add(word)
                logger.info(f"Loaded {len(stop_words) - count_before} non-Arabic words")
            except Exception as e:
                logger.warning(f"Error loading non-Arabic words: {e}")

        logger.info(f"Total stop words: {len(stop_words)}")
        return stop_words

    def _is_stop_word(self, lemma: str) -> bool:
        """Check if lemma is a stop word (including normalized forms)"""
        if lemma in self.stop_words:
            return True

        # Normalize and check again (remove diacritics for matching)
        normalized = lemma
        # Remove common Arabic diacritics
        for char in 'ًٌٍَُِّْٰۣۖۗۘۙۚۛۜ۟۠ۡۢۤۥۦۧۨ۩۪ۭ':
            normalized = normalized.replace(char, '')

        if normalized in self.stop_words:
            return True

        # Check without alef variations
        for alef in 'أإآٱ':
            normalized = normalized.replace(alef, 'ا')

        return normalized in self.stop_words


    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingGenerator()
        return self._embedder

    def load_lemmatized_files(self, limit: int = None, max_samples_per_lemma: int = 50):
        """
        Load lemmatized files with AGGRESSIVE memory management.

        Key strategies:
        1. Limit total lemmas in memory (cap at 50,000 most frequent)
        2. Only keep max_samples_per_lemma samples per lemma
        3. Skip corrupted files immediately
        4. Periodic garbage collection
        """
        logger.info(f"Loading lemmatized files from {self.lemmatized_dir}...")

        json_files = sorted(self.lemmatized_dir.glob('*_lemmatized.json'))

        if limit:
            json_files = json_files[:limit]

        self.global_stats['total_files'] = len(json_files)

        # First pass: just count lemma frequencies (very memory efficient)
        logger.info("Pass 1: Counting lemma frequencies...")
        lemma_counts: Counter = Counter()
        files_processed = 0
        corrupted_files = []

        for json_file in tqdm(json_files, desc="Counting"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for item in data.get('words_and_lemmas', []):
                    lemma = item.get('lemma', '').strip()
                    if lemma:
                        lemma_counts[lemma] += 1

                files_processed += 1
                del data

            except (json.JSONDecodeError, UnicodeDecodeError):
                corrupted_files.append(json_file.name)
            except Exception as e:
                logger.debug(f"Error counting {json_file.name}: {e}")

        if corrupted_files:
            logger.warning(f"Found {len(corrupted_files)} corrupted files (will skip)")

        # Keep only top N lemmas to process (memory cap)
        MAX_LEMMAS_TO_TRACK = 30000
        top_lemmas = set(l for l, _ in lemma_counts.most_common(MAX_LEMMAS_TO_TRACK))
        logger.info(f"Tracking top {len(top_lemmas):,} lemmas (out of {len(lemma_counts):,})")

        self.global_stats['total_words'] = sum(lemma_counts.values())
        self.global_stats['unique_lemmas'] = len(lemma_counts)

        # Store occurrence counts
        self.lemma_occurrence_counts = lemma_counts

        # Clear counter to save memory
        del lemma_counts
        gc.collect()

        # Second pass: collect samples for top lemmas only
        logger.info("Pass 2: Collecting form samples...")
        all_forms = set()

        for json_file in tqdm(json_files, desc="Sampling"):
            if json_file.name in corrupted_files:
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                words_and_lemmas = data.get('words_and_lemmas', [])

                for i, item in enumerate(words_and_lemmas):
                    word = item.get('word', '').strip()
                    lemma = item.get('lemma', '').strip()
                    pos = item.get('pos', 'UNKNOWN')

                    if not lemma or not word or pos in ['punc', 'digit']:
                        continue

                    # Only track top lemmas
                    if lemma not in top_lemmas:
                        continue

                    current_count = len(self.lemma_data[lemma])

                    if current_count < max_samples_per_lemma:
                        # Get minimal context
                        start = max(0, i - 2)
                        end = min(len(words_and_lemmas), i + 3)
                        context_words = [
                            w.get('word', '')
                            for w in words_and_lemmas[start:end]
                            if w.get('pos') not in ['punc', 'digit']
                        ]
                        context = ' '.join(context_words)

                        self.lemma_data[lemma].append((word, context, pos))
                        all_forms.add(word)
                    elif current_count < max_samples_per_lemma * 2:
                        # Probabilistic replacement for diversity
                        if random.random() < 0.05:  # 5% chance
                            idx = random.randint(0, max_samples_per_lemma - 1)
                            start = max(0, i - 2)
                            end = min(len(words_and_lemmas), i + 3)
                            context_words = [
                                w.get('word', '')
                                for w in words_and_lemmas[start:end]
                                if w.get('pos') not in ['punc', 'digit']
                            ]
                            context = ' '.join(context_words)
                            self.lemma_data[lemma][idx] = (word, context, pos)

                del data
                del words_and_lemmas

            except:
                pass  # Already logged in first pass

            # Periodic GC
            if files_processed % 1000 == 0:
                gc.collect()

        self.global_stats['unique_forms'] = len(all_forms)
        del all_forms
        gc.collect()

        logger.info(f"Loaded {files_processed} files (skipped {len(corrupted_files)} corrupted)")
        logger.info(f"Total words: {self.global_stats['total_words']:,}")
        logger.info(f"Unique lemmas tracked: {len(self.lemma_data):,}")
        logger.info(f"Samples per lemma: max {max_samples_per_lemma}")

    def _compute_intra_lemma_metrics(self, embeddings: np.ndarray,
                                      forms: List[str]) -> Tuple[float, float, float, float, List[str]]:
        """
        Compute metrics for spread within a single lemma.

        Returns:
            (mean_dist, max_dist, std_dist, centroid_spread, outlier_forms)
        """
        n = len(embeddings)

        if n < 2:
            return 0.0, 0.0, 0.0, 0.0, []

        # Compute pairwise distances (using cosine distance = 1 - similarity)
        similarities = cosine_similarity(embeddings)
        distances = 1 - similarities

        # Get upper triangle (exclude diagonal)
        upper_tri = distances[np.triu_indices(n, k=1)]

        mean_dist = float(np.mean(upper_tri))
        max_dist = float(np.max(upper_tri))
        std_dist = float(np.std(upper_tri)) if len(upper_tri) > 1 else 0.0

        # Compute centroid and distances to it
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        centroid_distances = 1 - cosine_similarity(embeddings, centroid).flatten()
        centroid_spread = float(np.mean(centroid_distances))

        # Find outliers (forms far from centroid)
        outlier_forms = []
        if n >= self.MIN_FORMS_FOR_OUTLIER:
            threshold = np.mean(centroid_distances) + 1.5 * np.std(centroid_distances)
            outlier_indices = np.where(centroid_distances > threshold)[0]
            outlier_forms = [forms[i] for i in outlier_indices]

        return mean_dist, max_dist, std_dist, centroid_spread, outlier_forms

    def _get_zipf_adjusted_threshold(self, frequency: int) -> float:
        """
        Adjust threshold based on Zipf's law.

        Frequent lemmas (like "و", "في") should have tighter clustering
        because they're more reliable. Rare lemmas can have more spread.
        """
        # Log-based adjustment
        if frequency >= 1000:
            return self.SPREAD_RATIO_GOOD * 0.8  # Stricter for very frequent
        elif frequency >= 100:
            return self.SPREAD_RATIO_GOOD * 0.9
        elif frequency >= 10:
            return self.SPREAD_RATIO_GOOD
        else:
            return self.SPREAD_RATIO_GOOD * 1.3  # More lenient for rare

    def process_lemmas_batch(self, lemmas: List[str],
                             all_centroids: Dict[str, np.ndarray]) -> List[LemmaSpreadStats]:
        """
        Process a batch of lemmas, compute their metrics.

        Steps:
        1. For each lemma, get unique forms with contexts
        2. Generate embeddings
        3. Compute intra-lemma spread
        4. Compute inter-lemma distance (to neighbors)
        5. Delete embeddings, keep only stats
        """
        results = []

        for lemma in lemmas:
            data = self.lemma_data[lemma]

            # Get unique forms with their best context
            form_contexts = {}
            pos_counts = Counter()

            for form, context, pos in data:
                if form not in form_contexts or len(context) > len(form_contexts[form]):
                    form_contexts[form] = context
                pos_counts[pos] += 1

            forms = list(form_contexts.keys())
            contexts = [form_contexts[f] for f in forms]

            # Create stats object
            stats = LemmaSpreadStats(
                lemma=lemma,
                total_forms=len(forms),
                total_occurrences=len(data),
                sample_forms=forms[:10],
                pos_distribution=dict(pos_counts)
            )

            # Need at least MIN_FORMS_FOR_SPREAD to compute spread
            if len(forms) < self.MIN_FORMS_FOR_SPREAD:
                stats.quality_status = "INSUFFICIENT_DATA"
                results.append(stats)
                continue

            try:
                # Generate embeddings for this lemma's forms
                embeddings = self.embedder.embed_texts(contexts)

                # Compute intra-lemma metrics
                (stats.intra_mean_distance,
                 stats.intra_max_distance,
                 stats.intra_std_distance,
                 stats.centroid_spread,
                 stats.outlier_forms) = self._compute_intra_lemma_metrics(embeddings, forms)

                # Compute centroid for this lemma
                centroid = np.mean(embeddings, axis=0)
                self.lemma_centroids[lemma] = centroid

                # Compute inter-lemma metrics (distance to other lemma centroids)
                if all_centroids:
                    other_lemmas = [l for l in all_centroids if l != lemma]
                    if other_lemmas:
                        other_centroids = np.array([all_centroids[l] for l in other_lemmas])

                        # Distance from this centroid to others
                        centroid_reshaped = centroid.reshape(1, -1)
                        distances = 1 - cosine_similarity(centroid_reshaped, other_centroids).flatten()

                        # Nearest neighbor
                        min_idx = np.argmin(distances)
                        stats.nearest_neighbor_distance = float(distances[min_idx])
                        stats.nearest_neighbor_lemma = other_lemmas[min_idx]

                        # Mean distance to k nearest
                        k = min(10, len(distances))
                        stats.mean_neighbor_distance = float(np.mean(np.sort(distances)[:k]))

                        # Compute spread ratio
                        if stats.nearest_neighbor_distance > 0:
                            stats.spread_ratio = stats.centroid_spread / stats.nearest_neighbor_distance

                        # Compute simple silhouette-like score
                        # (how much tighter is intra than inter)
                        if stats.mean_neighbor_distance > 0:
                            stats.silhouette_score = (stats.mean_neighbor_distance - stats.centroid_spread) / max(stats.mean_neighbor_distance, stats.centroid_spread)

                # Determine quality status based on Zipf-adjusted thresholds
                threshold_good = self._get_zipf_adjusted_threshold(stats.total_occurrences)
                threshold_suspicious = threshold_good * (self.SPREAD_RATIO_SUSPICIOUS / self.SPREAD_RATIO_GOOD)
                threshold_error = threshold_good * (self.SPREAD_RATIO_ERROR / self.SPREAD_RATIO_GOOD)

                if stats.spread_ratio < threshold_good:
                    stats.quality_status = "GOOD"
                    self.global_stats['good_lemmas'] += 1
                elif stats.spread_ratio < threshold_suspicious:
                    stats.quality_status = "ACCEPTABLE"
                elif stats.spread_ratio < threshold_error:
                    stats.quality_status = "SUSPICIOUS"
                    self.global_stats['suspicious_lemmas'] += 1
                else:
                    stats.quality_status = "LIKELY_ERROR"
                    self.global_stats['likely_errors'] += 1

                # Clean up embeddings for this lemma
                del embeddings

            except Exception as e:
                logger.warning(f"Error processing lemma '{lemma}': {e}")
                stats.quality_status = "ERROR"

            results.append(stats)

        return results

    def run_validation(self, batch_size: int = 100,
                       max_lemmas: int = None,
                       min_forms: int = 2) -> pd.DataFrame:
        """
        Run the full validation pipeline.

        Args:
            batch_size: Number of lemmas to process before cleaning memory
            max_lemmas: Maximum lemmas to process (for testing)
            min_forms: Minimum forms a lemma must have to be processed
        """
        start_time = datetime.now()

        # Filter lemmas by minimum forms
        valid_lemmas = [
            lemma for lemma, data in self.lemma_data.items()
            if len(set(d[0] for d in data)) >= min_forms
        ]

        # Sort by frequency (Zipf's law - process most frequent first)
        valid_lemmas.sort(key=lambda l: -len(self.lemma_data[l]))

        if max_lemmas:
            valid_lemmas = valid_lemmas[:max_lemmas]

        logger.info(f"Processing {len(valid_lemmas)} lemmas with {min_forms}+ forms...")

        # Filter out stop words
        filtered_lemmas = [l for l in valid_lemmas if not self._is_stop_word(l)]
        stop_word_count = len(valid_lemmas) - len(filtered_lemmas)
        self.global_stats['lemmas_filtered_stopwords'] = stop_word_count

        if stop_word_count > 0:
            logger.info(f"Filtered out {stop_word_count} stop word lemmas")
            logger.info(f"Remaining: {len(filtered_lemmas)} content lemmas")

        valid_lemmas = filtered_lemmas

        # First pass: compute centroids for all lemmas
        logger.info("Pass 1: Computing lemma centroids...")

        for i in tqdm(range(0, len(valid_lemmas), batch_size), desc="Computing centroids"):
            batch = valid_lemmas[i:i + batch_size]

            for lemma in batch:
                data = self.lemma_data[lemma]

                # Get unique forms with contexts
                form_contexts = {}
                for form, context, pos in data:
                    if form not in form_contexts:
                        form_contexts[form] = context

                contexts = list(form_contexts.values())

                if len(contexts) >= self.MIN_FORMS_FOR_SPREAD:
                    try:
                        embeddings = self.embedder.embed_texts(contexts)
                        self.lemma_centroids[lemma] = np.mean(embeddings, axis=0)
                        del embeddings
                    except Exception as e:
                        logger.warning(f"Error computing centroid for '{lemma}': {e}")

            # Clean up memory after each batch
            self.embedder.cleanup()

        logger.info(f"Computed {len(self.lemma_centroids)} centroids")

        # Second pass: compute full metrics with neighbor information
        logger.info("Pass 2: Computing spread metrics...")

        all_stats = []

        for i in tqdm(range(0, len(valid_lemmas), batch_size), desc="Processing lemmas"):
            batch = valid_lemmas[i:i + batch_size]
            batch_stats = self.process_lemmas_batch(batch, self.lemma_centroids)
            all_stats.extend(batch_stats)

            # Clean up memory
            self.embedder.cleanup()
            gc.collect()

        # Store results
        for stats in all_stats:
            self.lemma_stats[stats.lemma] = stats

        # Compute relative quality scores based on corpus statistics
        # This compares each lemma to the corpus average
        spread_ratios = [s.spread_ratio for s in all_stats if s.spread_ratio > 0]
        if spread_ratios:
            corpus_mean = np.mean(spread_ratios)
            corpus_std = np.std(spread_ratios)

            logger.info(f"Corpus spread ratio: mean={corpus_mean:.3f}, std={corpus_std:.3f}")

            # Recategorize based on relative position in distribution
            self.global_stats['good_lemmas'] = 0
            self.global_stats['suspicious_lemmas'] = 0
            self.global_stats['likely_errors'] = 0

            for stats in all_stats:
                if stats.spread_ratio == 0:
                    continue

                # Z-score relative to corpus
                z_score = (stats.spread_ratio - corpus_mean) / corpus_std if corpus_std > 0 else 0

                # Categorize based on z-score
                # Within 1 std of mean = GOOD
                # 1-2 std above mean = SUSPICIOUS (considering frequency)
                # 2+ std above mean = LIKELY_ERROR

                # Frequency adjustment: high-frequency lemmas should be tighter
                freq_factor = 1.0
                if stats.total_occurrences >= 500:
                    freq_factor = 0.7  # Stricter for very frequent
                elif stats.total_occurrences >= 100:
                    freq_factor = 0.85
                elif stats.total_occurrences <= 10:
                    freq_factor = 1.3  # More lenient for rare

                adjusted_z = z_score * freq_factor

                if adjusted_z <= 1.0:
                    stats.quality_status = "GOOD"
                    self.global_stats['good_lemmas'] += 1
                elif adjusted_z <= 1.5:
                    stats.quality_status = "ACCEPTABLE"
                elif adjusted_z <= 2.0:
                    stats.quality_status = "SUSPICIOUS"
                    self.global_stats['suspicious_lemmas'] += 1
                else:
                    stats.quality_status = "LIKELY_ERROR"
                    self.global_stats['likely_errors'] += 1

        # Create DataFrame
        df = self._stats_to_dataframe(all_stats)

        # Compute processing time
        self.global_stats['processing_time'] = (datetime.now() - start_time).total_seconds()

        # Clean up embedder
        if self._embedder is not None:
            self._embedder.cleanup()
            del self._embedder
            self._embedder = None

        # Clear centroids to free memory
        self.lemma_centroids.clear()
        gc.collect()

        return df

    def _stats_to_dataframe(self, stats_list: List[LemmaSpreadStats]) -> pd.DataFrame:
        """Convert stats list to DataFrame"""
        rows = []

        for s in stats_list:
            rows.append({
                'lemma': s.lemma,
                'total_forms': s.total_forms,
                'total_occurrences': s.total_occurrences,
                'intra_mean_distance': round(s.intra_mean_distance, 4),
                'intra_max_distance': round(s.intra_max_distance, 4),
                'intra_std_distance': round(s.intra_std_distance, 4),
                'centroid_spread': round(s.centroid_spread, 4),
                'nearest_neighbor_distance': round(s.nearest_neighbor_distance, 4),
                'nearest_neighbor_lemma': s.nearest_neighbor_lemma,
                'mean_neighbor_distance': round(s.mean_neighbor_distance, 4),
                'spread_ratio': round(s.spread_ratio, 4),
                'silhouette_score': round(s.silhouette_score, 4),
                'quality_status': s.quality_status,
                'num_pos_tags': len(s.pos_distribution),
                'pos_distribution': str(s.pos_distribution),
                'sample_forms': ', '.join(s.sample_forms[:5]),
                'outlier_forms': ', '.join(s.outlier_forms[:5]),
            })

        df = pd.DataFrame(rows)
        return df

    def generate_reports(self, df: pd.DataFrame):
        """Generate validation reports"""
        logger.info("Generating reports...")

        # 1. Full results
        full_report = self.output_dir / 'lemma_spread_full.csv'
        df.to_csv(full_report, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Full results: {full_report}")

        # 2. Likely errors (highest spread ratio)
        errors_df = df[df['quality_status'] == 'LIKELY_ERROR'].sort_values(
            'spread_ratio', ascending=False
        )
        if not errors_df.empty:
            errors_file = self.output_dir / 'likely_errors.csv'
            errors_df.to_csv(errors_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Likely errors: {errors_file} ({len(errors_df)} cases)")

        # 3. Suspicious cases
        suspicious_df = df[df['quality_status'] == 'SUSPICIOUS'].sort_values(
            'spread_ratio', ascending=False
        )
        if not suspicious_df.empty:
            suspicious_file = self.output_dir / 'suspicious_lemmas.csv'
            suspicious_df.to_csv(suspicious_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Suspicious: {suspicious_file} ({len(suspicious_df)} cases)")

        # 4. Good lemmas (for reference)
        good_df = df[df['quality_status'] == 'GOOD'].sort_values(
            'total_occurrences', ascending=False
        )
        if not good_df.empty:
            good_file = self.output_dir / 'good_lemmas.csv'
            good_df.head(500).to_csv(good_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Good lemmas (top 500): {good_file}")

        # 5. Lemmas with outlier forms
        outliers_df = df[df['outlier_forms'].str.len() > 0].sort_values(
            'total_occurrences', ascending=False
        )
        if not outliers_df.empty:
            outliers_file = self.output_dir / 'lemmas_with_outliers.csv'
            outliers_df.to_csv(outliers_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Lemmas with outliers: {outliers_file} ({len(outliers_df)} cases)")

        # 6. Summary statistics
        summary = {
            **self.global_stats,
            'lemmas_processed': len(df),
            'quality_distribution': df['quality_status'].value_counts().to_dict(),
            'spread_ratio_stats': {
                'mean': round(df['spread_ratio'].mean(), 4),
                'median': round(df['spread_ratio'].median(), 4),
                'std': round(df['spread_ratio'].std(), 4),
                'min': round(df['spread_ratio'].min(), 4),
                'max': round(df['spread_ratio'].max(), 4),
            },
            'thresholds': {
                'good': self.SPREAD_RATIO_GOOD,
                'suspicious': self.SPREAD_RATIO_SUSPICIOUS,
                'error': self.SPREAD_RATIO_ERROR,
            }
        }

        summary_file = self.output_dir / 'spread_validation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Summary: {summary_file}")

        # Print summary
        self._print_summary(df)

    def _print_summary(self, df: pd.DataFrame):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("LEMMA SPREAD VALIDATION REPORT")
        print("=" * 80)

        print(f"\n📊 Data Statistics:")
        print(f"   Files analyzed:     {self.global_stats['total_files']:,}")
        print(f"   Total words:        {self.global_stats['total_words']:,}")
        print(f"   Unique lemmas:      {self.global_stats['unique_lemmas']:,}")
        print(f"   Lemmas processed:   {len(df):,}")
        if self.global_stats.get('stop_words_loaded', 0) > 0:
            print(f"   Stop words loaded:  {self.global_stats['stop_words_loaded']:,}")
            print(f"   Lemmas filtered:    {self.global_stats['lemmas_filtered_stopwords']:,}")

        print(f"\n📈 Quality Distribution:")
        for status, count in df['quality_status'].value_counts().items():
            pct = count / len(df) * 100
            bar = '█' * int(pct / 2)
            print(f"   {status:<18} {count:>6} ({pct:>5.1f}%) {bar}")

        print(f"\n📏 Spread Ratio Statistics:")
        print(f"   Mean:   {df['spread_ratio'].mean():.4f}")
        print(f"   Median: {df['spread_ratio'].median():.4f}")
        print(f"   Std:    {df['spread_ratio'].std():.4f}")
        print(f"   Min:    {df['spread_ratio'].min():.4f}")
        print(f"   Max:    {df['spread_ratio'].max():.4f}")

        # Top errors
        errors = df[df['quality_status'] == 'LIKELY_ERROR'].head(15)
        if not errors.empty:
            print(f"\n🔴 Top 15 Likely Errors (highest spread ratio):")
            print("-" * 100)
            print(f"{'Lemma':<20}{'Forms':>6}{'Occurrences':>12}{'Spread':>10}{'Nearest':>10}{'Neighbor':<20}{'Sample Forms'}")
            print("-" * 100)
            for _, row in errors.iterrows():
                print(f"{row['lemma']:<20}{row['total_forms']:>6}{row['total_occurrences']:>12}{row['spread_ratio']:>10.3f}{row['nearest_neighbor_distance']:>10.3f}{row['nearest_neighbor_lemma']:<20}{row['sample_forms'][:40]}")

        # Suspicious with many forms (high impact)
        suspicious_high = df[(df['quality_status'] == 'SUSPICIOUS') & (df['total_occurrences'] > 50)]
        if not suspicious_high.empty:
            print(f"\n🟡 Top Suspicious High-Frequency Lemmas:")
            print("-" * 100)
            for _, row in suspicious_high.head(10).iterrows():
                print(f"{row['lemma']:<20}{row['total_forms']:>6}{row['total_occurrences']:>12}{row['spread_ratio']:>10.3f}  {row['sample_forms'][:50]}")

        # Lemmas with outliers
        with_outliers = df[df['outlier_forms'].str.len() > 0].head(10)
        if not with_outliers.empty:
            print(f"\n⚠️ Lemmas with Outlier Forms:")
            print("-" * 80)
            for _, row in with_outliers.iterrows():
                print(f"{row['lemma']:<20} Outliers: {row['outlier_forms'][:60]}")

        print(f"\n⏱️ Processing time: {self.global_stats['processing_time']:.1f} seconds")
        print(f"\n✅ Reports saved to: {self.output_dir}/")
        print("=" * 80)

    def run(self, file_limit: int = None, max_lemmas: int = None,
            min_forms: int = 3, batch_size: int = 100, max_samples: int = 20):
        """
        Run the full validation pipeline.

        Args:
            file_limit: Max files to load
            max_lemmas: Max lemmas to process
            min_forms: Minimum forms for a lemma to be processed
            batch_size: Batch size for memory management
            max_samples: Max samples per lemma (for memory efficiency)
        """
        # Step 1: Load data
        self.load_lemmatized_files(limit=file_limit, max_samples_per_lemma=max_samples)

        if not self.lemma_data:
            logger.error("No data loaded!")
            return

        # Step 2: Run validation
        df = self.run_validation(
            batch_size=batch_size,
            max_lemmas=max_lemmas,
            min_forms=min_forms
        )

        # Step 3: Generate reports
        self.generate_reports(df)


def main():
    parser = argparse.ArgumentParser(
        description='Validate lemmatization by analyzing embedding spread'
    )
    parser.add_argument('--input', '-i', default=None,
                        help='Directory with lemmatized JSON files')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory for reports')
    parser.add_argument('--stop-words', '-s', default=None,
                        help='Directory with stop words files')
    parser.add_argument('--file-limit', type=int, default=None,
                        help='Limit number of files to process')
    parser.add_argument('--max-lemmas', type=int, default=None,
                        help='Maximum lemmas to process')
    parser.add_argument('--min-forms', type=int, default=3,
                        help='Minimum forms for a lemma to be analyzed')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for memory management')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='Max samples per lemma (reduces RAM usage, default: 20)')
    parser.add_argument('--no-filter-stopwords', action='store_true',
                        help='Disable stop word filtering')

    args = parser.parse_args()

    validator = LemmaSpreadValidator(
        lemmatized_dir=args.input,
        output_dir=args.output,
        stop_words_dir=args.stop_words if not args.no_filter_stopwords else "DISABLED"
    )

    validator.run(
        file_limit=args.file_limit,
        max_lemmas=args.max_lemmas,
        min_forms=args.min_forms,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )


if __name__ == '__main__':
    main()

