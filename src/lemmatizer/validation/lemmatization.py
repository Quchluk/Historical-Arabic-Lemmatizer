#!/usr/bin/env python3
"""
Lemmatization Validator using TF (Term Frequency) + AraBERT Embeddings
=======================================================================
Validates CAMeL Tools lemmatization by analyzing:
1. TF Analysis: Distribution of word forms per lemma across POS tags
2. Embedding Similarity: AraBERT contextual embeddings

TF-Based Validation Logic:
- If a word form appears under the SAME lemma with DIFFERENT POS tags:
  → Check if the form distribution is consistent
  → Forms that appear predominantly with one POS but occasionally with another = likely error
- Form Entropy: High entropy in POS distribution per form = genuine polysemy
                Low entropy (one POS dominates) = potential tagging inconsistency

Combined with Embeddings:
- Same lemma + different POS + HIGH embedding similarity → Likely POS error
- Same lemma + different POS + LOW embedding similarity → Genuine polysemy (correct)
"""

import json
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
from sklearn.metrics.pairwise import cosine_similarity
import gc
import math
from lemmatizer.config import LOGS_DIR, DATA_DIR, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class WordInstance:
    """A single occurrence of a word with its context"""
    word: str
    lemma: str
    pos: str
    context: str  # Surrounding words
    source_file: str
    position: int  # Position in the source file


@dataclass
class FormStatistics:
    """Statistics for a single word form under a lemma"""
    form: str
    lemma: str
    pos_distribution: Dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    dominant_pos: str = ""
    pos_entropy: float = 0.0
    is_consistent: bool = True  # True if form almost always has same POS
    consistency_score: float = 1.0  # 1.0 = always same POS, 0.0 = equally distributed


class TFValidator:
    """
    Term Frequency based validator for lemmatization.

    Analyzes the distribution of word forms per lemma to detect:
    1. Forms that inconsistently switch between POS tags (potential errors)
    2. Forms with genuine polysemy (different meanings → different POS)
    """

    def __init__(self):
        # Storage: lemma -> form -> pos -> count
        self.lemma_form_pos: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        # Storage: lemma -> form -> list of contexts
        self.form_contexts: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        # Form statistics cache
        self.form_stats: Dict[str, Dict[str, FormStatistics]] = {}

    def add_instance(self, word: str, lemma: str, pos: str, context: str = ""):
        """Add a word instance to the TF analyzer"""
        self.lemma_form_pos[lemma][word][pos] += 1
        if context and len(self.form_contexts[lemma][word]) < 10:
            self.form_contexts[lemma][word].append(context)

    def calculate_entropy(self, distribution: Dict[str, int]) -> float:
        """
        Calculate Shannon entropy of a distribution.
        Higher entropy = more evenly distributed (genuine polysemy)
        Lower entropy = one value dominates (potential error or consistent tagging)
        """
        total = sum(distribution.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def calculate_consistency_score(self, distribution: Dict[str, int]) -> float:
        """
        Calculate how consistently a form is tagged with one POS.
        1.0 = always the same POS
        0.0 = equally distributed across all POS tags
        """
        total = sum(distribution.values())
        if total == 0:
            return 1.0

        max_count = max(distribution.values())
        return max_count / total

    def analyze_forms(self) -> Dict[str, Dict[str, FormStatistics]]:
        """
        Analyze all forms and compute statistics.

        Returns:
            Dict[lemma] -> Dict[form] -> FormStatistics
        """
        logger.info("Analyzing word form distributions...")

        self.form_stats = {}

        for lemma, forms in tqdm(self.lemma_form_pos.items(), desc="Analyzing forms"):
            self.form_stats[lemma] = {}

            for form, pos_counts in forms.items():
                total = sum(pos_counts.values())
                dominant_pos = pos_counts.most_common(1)[0][0] if pos_counts else ""
                entropy = self.calculate_entropy(pos_counts)
                consistency = self.calculate_consistency_score(pos_counts)

                # A form is "inconsistent" if it has multiple POS with significant counts
                # (not just 1 or 2 outliers)
                is_consistent = consistency >= 0.9 or total < 5

                stats = FormStatistics(
                    form=form,
                    lemma=lemma,
                    pos_distribution=dict(pos_counts),
                    total_count=total,
                    dominant_pos=dominant_pos,
                    pos_entropy=entropy,
                    is_consistent=is_consistent,
                    consistency_score=consistency
                )

                self.form_stats[lemma][form] = stats

        return self.form_stats

    def find_inconsistent_forms(self, min_count: int = 5, max_consistency: float = 0.85) -> List[FormStatistics]:
        """
        Find forms that are tagged inconsistently (potential errors).

        Args:
            min_count: Minimum total occurrences to consider
            max_consistency: Forms with consistency below this are flagged

        Returns:
            List of FormStatistics for inconsistent forms
        """
        if not self.form_stats:
            self.analyze_forms()

        inconsistent = []

        for lemma, forms in self.form_stats.items():
            for form, stats in forms.items():
                if stats.total_count >= min_count and stats.consistency_score < max_consistency:
                    inconsistent.append(stats)

        # Sort by inconsistency (lower consistency = more inconsistent)
        inconsistent.sort(key=lambda x: (x.consistency_score, -x.total_count))

        return inconsistent

    def find_potential_errors(self, min_count: int = 3, error_threshold: float = 0.15) -> List[Dict]:
        """
        Find potential tagging errors using TF analysis.

        Logic: If a form appears N times with POS-A and only 1-2 times with POS-B,
        and N >> 1-2, then POS-B occurrences are likely errors.

        Args:
            min_count: Minimum occurrences of dominant POS
            error_threshold: If minority POS < threshold * dominant, it's flagged

        Returns:
            List of potential errors with details
        """
        if not self.form_stats:
            self.analyze_forms()

        potential_errors = []

        for lemma, forms in self.form_stats.items():
            for form, stats in forms.items():
                if len(stats.pos_distribution) < 2:
                    continue

                # Sort POS by frequency
                sorted_pos = sorted(stats.pos_distribution.items(), key=lambda x: -x[1])
                dominant_pos, dominant_count = sorted_pos[0]

                if dominant_count < min_count:
                    continue

                # Check for minority POS that might be errors
                for minority_pos, minority_count in sorted_pos[1:]:
                    ratio = minority_count / dominant_count

                    if ratio < error_threshold:
                        # This looks like an error - minority POS appears rarely
                        potential_errors.append({
                            'lemma': lemma,
                            'form': form,
                            'dominant_pos': dominant_pos,
                            'dominant_count': dominant_count,
                            'minority_pos': minority_pos,
                            'minority_count': minority_count,
                            'error_ratio': ratio,
                            'total_count': stats.total_count,
                            'confidence': 1.0 - ratio,
                            'sample_contexts': self.form_contexts.get(lemma, {}).get(form, [])[:3]
                        })

        # Sort by confidence
        potential_errors.sort(key=lambda x: -x['confidence'])

        return potential_errors

    def find_lemma_pos_conflicts(self, min_total_count: int = 10, min_minority_ratio: float = 0.05) -> List[Dict]:
        """
        Find lemmas where the SAME LEMMA has different POS tags across ALL its forms.

        This is the key TF-based analysis: if a lemma appears as noun 95% of times
        but verb 5% of times, the 5% might be errors OR genuine polysemy.

        Args:
            min_total_count: Minimum total occurrences of the lemma
            min_minority_ratio: Minimum ratio for a POS to be considered (filter noise)

        Returns:
            List of lemma conflicts with statistics
        """
        if not self.form_stats:
            self.analyze_forms()

        conflicts = []

        for lemma, forms in self.form_stats.items():
            # Aggregate POS counts across ALL forms of this lemma
            lemma_pos_total = Counter()
            all_forms_list = []

            for form, stats in forms.items():
                for pos, count in stats.pos_distribution.items():
                    lemma_pos_total[pos] += count
                all_forms_list.append(form)

            total_count = sum(lemma_pos_total.values())

            if total_count < min_total_count:
                continue

            if len(lemma_pos_total) < 2:
                continue

            # Sort by frequency
            sorted_pos = sorted(lemma_pos_total.items(), key=lambda x: -x[1])
            dominant_pos, dominant_count = sorted_pos[0]

            # Check each minority POS
            for minority_pos, minority_count in sorted_pos[1:]:
                ratio = minority_count / total_count

                if ratio < min_minority_ratio:
                    continue  # Too rare to be meaningful

                # Get sample forms for each POS
                forms_with_dominant = []
                forms_with_minority = []

                for form, stats in forms.items():
                    if dominant_pos in stats.pos_distribution:
                        forms_with_dominant.append((form, stats.pos_distribution[dominant_pos]))
                    if minority_pos in stats.pos_distribution:
                        forms_with_minority.append((form, stats.pos_distribution[minority_pos]))

                # Sort by count
                forms_with_dominant.sort(key=lambda x: -x[1])
                forms_with_minority.sort(key=lambda x: -x[1])

                # Get sample contexts
                sample_contexts_dominant = []
                sample_contexts_minority = []
                for form, _ in forms_with_dominant[:3]:
                    ctxs = self.form_contexts.get(lemma, {}).get(form, [])
                    if ctxs:
                        sample_contexts_dominant.append(ctxs[0])
                for form, _ in forms_with_minority[:3]:
                    ctxs = self.form_contexts.get(lemma, {}).get(form, [])
                    if ctxs:
                        sample_contexts_minority.append(ctxs[0])

                # Determine if this looks like error or polysemy
                # Error indicators: minority ratio is very small, minority forms are rare
                # Polysemy indicators: both POS have substantial counts, multiple forms

                is_likely_error = ratio < 0.15 and len(forms_with_minority) <= 2
                is_likely_polysemy = ratio >= 0.25 or len(forms_with_minority) >= 3

                status = 'LIKELY_ERROR' if is_likely_error else ('LIKELY_POLYSEMY' if is_likely_polysemy else 'UNCERTAIN')

                conflicts.append({
                    'lemma': lemma,
                    'dominant_pos': dominant_pos,
                    'dominant_count': dominant_count,
                    'dominant_ratio': round(dominant_count / total_count, 4),
                    'minority_pos': minority_pos,
                    'minority_count': minority_count,
                    'minority_ratio': round(ratio, 4),
                    'total_count': total_count,
                    'total_forms': len(all_forms_list),
                    'forms_with_dominant': len(forms_with_dominant),
                    'forms_with_minority': len(forms_with_minority),
                    'sample_dominant_forms': ', '.join([f[0] for f in forms_with_dominant[:5]]),
                    'sample_minority_forms': ', '.join([f[0] for f in forms_with_minority[:5]]),
                    'sample_context_dominant': sample_contexts_dominant[0] if sample_contexts_dominant else '',
                    'sample_context_minority': sample_contexts_minority[0] if sample_contexts_minority else '',
                    'status': status,
                    'confidence': 1.0 - ratio if is_likely_error else ratio
                })

        # Sort by total count (most frequent lemmas first)
        conflicts.sort(key=lambda x: -x['total_count'])

        return conflicts

    def find_genuine_polysemy(self, min_count: int = 5, min_entropy: float = 0.8) -> List[FormStatistics]:
        """
        Find forms that genuinely have multiple meanings (polysemy).

        Logic: High entropy in POS distribution with sufficient counts
        indicates genuine polysemy rather than tagging errors.

        Args:
            min_count: Minimum total occurrences
            min_entropy: Minimum entropy to be considered polysemous

        Returns:
            List of polysemous forms
        """
        if not self.form_stats:
            self.analyze_forms()

        polysemous = []

        for lemma, forms in self.form_stats.items():
            for form, stats in forms.items():
                if (stats.total_count >= min_count and
                    len(stats.pos_distribution) >= 2 and
                    stats.pos_entropy >= min_entropy):
                    polysemous.append(stats)

        # Sort by entropy (higher = more polysemous)
        polysemous.sort(key=lambda x: -x.pos_entropy)

        return polysemous

    def get_lemma_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all lemmas with their form statistics.
        """
        if not self.form_stats:
            self.analyze_forms()

        rows = []
        for lemma, forms in self.form_stats.items():
            total_forms = len(forms)
            total_occurrences = sum(s.total_count for s in forms.values())

            # Aggregate POS distribution
            lemma_pos = Counter()
            for stats in forms.values():
                for pos, count in stats.pos_distribution.items():
                    lemma_pos[pos] += count

            # Count inconsistent forms
            inconsistent_forms = sum(1 for s in forms.values() if not s.is_consistent)

            # Average consistency
            avg_consistency = np.mean([s.consistency_score for s in forms.values()]) if forms else 1.0

            rows.append({
                'lemma': lemma,
                'total_forms': total_forms,
                'total_occurrences': total_occurrences,
                'num_pos_tags': len(lemma_pos),
                'pos_distribution': str(dict(lemma_pos)),
                'inconsistent_forms': inconsistent_forms,
                'avg_consistency': round(avg_consistency, 4),
                'dominant_pos': lemma_pos.most_common(1)[0][0] if lemma_pos else ''
            })

        df = pd.DataFrame(rows)
        df.sort_values('total_occurrences', ascending=False, inplace=True)

        return df


class AraBERTValidator:
    """Generate embeddings for validation using AraBERT"""

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

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def get_word_embedding(self, word: str, context: str) -> np.ndarray:
        """Get embedding for a word in its context."""
        text = context if word in context else f"{context} {word}"

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze()

        return embedding.cpu().numpy()

    def batch_embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Batch embed multiple texts for efficiency."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=128,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class LemmatizationValidator:
    """Main validator combining TF analysis and embedding validation"""

    def __init__(self, lemmatized_dir: str = None, output_dir: str = None):
        self.lemmatized_dir = Path(lemmatized_dir) if lemmatized_dir else PROCESSED_DATA_DIR / "camel_lemmatized"
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR / "validation_reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TF Validator
        self.tf_validator = TFValidator()

        # Storage for word instances grouped by lemma
        self.lemma_instances: Dict[str, List[WordInstance]] = defaultdict(list)

        # Statistics
        self.stats = {
            'total_files': 0,
            'total_words': 0,
            'unique_lemmas': 0,
            'unique_forms': 0,
            'lemmas_with_pos_conflicts': 0,
            'tf_potential_errors': 0,
            'tf_genuine_polysemy': 0,
            'embedding_suspected_errors': 0,
            'embedding_confirmed_polysemy': 0,
            'uncertain_cases': 0
        }

        # Thresholds
        self.HIGH_SIMILARITY_THRESHOLD = 0.85
        self.LOW_SIMILARITY_THRESHOLD = 0.60
        self.TF_ERROR_THRESHOLD = 0.25  # Below this ratio = likely error (more sensitive)
        self.TF_POLYSEMY_MIN_ENTROPY = 0.5  # Lower entropy threshold for polysemy detection

        # Embedder (lazy initialization)
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            self._embedder = AraBERTValidator()
        return self._embedder

    def load_lemmatized_files(self, limit: int = None):
        """Load all lemmatized JSON files"""
        logger.info(f"Loading lemmatized files from {self.lemmatized_dir}...")

        json_files = sorted(self.lemmatized_dir.glob('*_lemmatized.json'))

        if limit:
            json_files = json_files[:limit]

        self.stats['total_files'] = len(json_files)

        for json_file in tqdm(json_files, desc="Loading files"):
            self._load_single_file(json_file)

        self.stats['unique_lemmas'] = len(self.lemma_instances)
        self.stats['unique_forms'] = sum(
            len(forms) for forms in self.tf_validator.lemma_form_pos.values()
        )
        logger.info(f"Loaded {self.stats['total_words']:,} words")
        logger.info(f"Found {self.stats['unique_lemmas']:,} unique lemmas")
        logger.info(f"Found {self.stats['unique_forms']:,} unique word forms")

    def load_lemmatized_files_by_names(self, file_names: list):
        """Load specific lemmatized JSON files by their names"""
        logger.info(f"Loading {len(file_names)} specific files from {self.lemmatized_dir}...")

        loaded = 0
        for file_name in tqdm(file_names, desc="Loading files"):
            json_file = self.lemmatized_dir / file_name
            if json_file.exists():
                self._load_single_file(json_file)
                loaded += 1
            else:
                logger.warning(f"File not found: {file_name}")

        self.stats['total_files'] = loaded
        self.stats['unique_lemmas'] = len(self.lemma_instances)
        self.stats['unique_forms'] = sum(
            len(forms) for forms in self.tf_validator.lemma_form_pos.values()
        )
        logger.info(f"Loaded {loaded} files, {self.stats['total_words']:,} words")
        logger.info(f"Found {self.stats['unique_lemmas']:,} unique lemmas")
        logger.info(f"Found {self.stats['unique_forms']:,} unique word forms")

    def _load_single_file(self, json_file: Path):
        """Load a single lemmatized JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            words_and_lemmas = data.get('words_and_lemmas', [])
            source_file = data.get('source_file', str(json_file))

            for i, item in enumerate(words_and_lemmas):
                word = item.get('word', '')
                lemma = item.get('lemma', '')
                pos = item.get('pos', 'UNKNOWN')

                if pos in ['punc', 'digit'] or not lemma:
                    continue

                # Get context
                context_start = max(0, i - 5)
                context_end = min(len(words_and_lemmas), i + 6)
                context_words = [
                    w.get('word', '')
                    for w in words_and_lemmas[context_start:context_end]
                    if w.get('pos') not in ['punc', 'digit']
                ]
                context = ' '.join(context_words)

                # Add to TF validator
                self.tf_validator.add_instance(word, lemma, pos, context)

                # Create word instance
                instance = WordInstance(
                    word=word,
                    lemma=lemma,
                    pos=pos,
                    context=context,
                    source_file=source_file,
                    position=i
                )

                self.lemma_instances[lemma].append(instance)
                self.stats['total_words'] += 1

        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")

    def run_tf_analysis(self) -> Tuple[List[Dict], List[FormStatistics], List[Dict]]:
        """
        Run TF-based validation to find errors and polysemy.

        Returns:
            (potential_errors, genuine_polysemy, lemma_conflicts)
        """
        logger.info("Running TF-based analysis...")

        # Analyze forms
        self.tf_validator.analyze_forms()

        # Find form-level potential errors (same form, different POS)
        potential_errors = self.tf_validator.find_potential_errors(
            min_count=3,
            error_threshold=self.TF_ERROR_THRESHOLD
        )
        self.stats['tf_potential_errors'] = len(potential_errors)
        logger.info(f"Found {len(potential_errors)} form-level TF-based errors")

        # Find genuine polysemy at form level
        genuine_polysemy = self.tf_validator.find_genuine_polysemy(
            min_count=5,
            min_entropy=self.TF_POLYSEMY_MIN_ENTROPY
        )
        self.stats['tf_genuine_polysemy'] = len(genuine_polysemy)
        logger.info(f"Found {len(genuine_polysemy)} form-level polysemy cases")

        # Find lemma-level POS conflicts (same lemma, different POS across all forms)
        lemma_conflicts = self.tf_validator.find_lemma_pos_conflicts(
            min_total_count=10,
            min_minority_ratio=0.05
        )
        self.stats['tf_lemma_conflicts'] = len(lemma_conflicts)
        logger.info(f"Found {len(lemma_conflicts)} lemma-level POS conflicts")

        return potential_errors, genuine_polysemy, lemma_conflicts

    def find_pos_conflicts(self) -> Dict[str, Dict[str, List[WordInstance]]]:
        """Find lemmas that have multiple POS tags."""
        logger.info("Finding POS conflicts...")

        conflicts = {}

        for lemma, instances in self.lemma_instances.items():
            pos_groups = defaultdict(list)
            for inst in instances:
                pos_groups[inst.pos].append(inst)

            if len(pos_groups) > 1:
                conflicts[lemma] = dict(pos_groups)

        self.stats['lemmas_with_pos_conflicts'] = len(conflicts)
        logger.info(f"Found {len(conflicts):,} lemmas with POS conflicts")

        return conflicts

    def validate_with_embeddings(self, conflicts: Dict[str, Dict[str, List[WordInstance]]],
                                 max_samples_per_pos: int = 5) -> pd.DataFrame:
        """Validate POS conflicts using embedding similarity."""
        logger.info("Validating conflicts with embeddings...")

        results = []

        for lemma, pos_groups in tqdm(conflicts.items(), desc="Validating"):
            pos_list = list(pos_groups.keys())

            for i in range(len(pos_list)):
                for j in range(i + 1, len(pos_list)):
                    pos1, pos2 = pos_list[i], pos_list[j]

                    instances1 = pos_groups[pos1][:max_samples_per_pos]
                    instances2 = pos_groups[pos2][:max_samples_per_pos]

                    contexts1 = [inst.context for inst in instances1]
                    contexts2 = [inst.context for inst in instances2]

                    try:
                        embeddings1 = self.embedder.batch_embed(contexts1)
                        embeddings2 = self.embedder.batch_embed(contexts2)

                        similarities = cosine_similarity(embeddings1, embeddings2)
                        avg_similarity = np.mean(similarities)
                        max_similarity = np.max(similarities)
                        min_similarity = np.min(similarities)

                        if avg_similarity > self.HIGH_SIMILARITY_THRESHOLD:
                            status = 'LIKELY_ERROR'
                            self.stats['embedding_suspected_errors'] += 1
                        elif avg_similarity < self.LOW_SIMILARITY_THRESHOLD:
                            status = 'GENUINE_POLYSEMY'
                            self.stats['embedding_confirmed_polysemy'] += 1
                        else:
                            status = 'UNCERTAIN'
                            self.stats['uncertain_cases'] += 1

                        # Get TF stats for this lemma
                        tf_stats = self.tf_validator.form_stats.get(lemma, {})
                        forms_with_both_pos = []
                        for form, stats in tf_stats.items():
                            if pos1 in stats.pos_distribution and pos2 in stats.pos_distribution:
                                forms_with_both_pos.append({
                                    'form': form,
                                    'pos1_count': stats.pos_distribution[pos1],
                                    'pos2_count': stats.pos_distribution[pos2],
                                    'consistency': stats.consistency_score
                                })

                        sample_words1 = list(set(inst.word for inst in instances1))[:3]
                        sample_words2 = list(set(inst.word for inst in instances2))[:3]

                        results.append({
                            'lemma': lemma,
                            'pos1': pos1,
                            'pos2': pos2,
                            'count_pos1': len(pos_groups[pos1]),
                            'count_pos2': len(pos_groups[pos2]),
                            'avg_similarity': round(avg_similarity, 4),
                            'max_similarity': round(max_similarity, 4),
                            'min_similarity': round(min_similarity, 4),
                            'status': status,
                            'tf_forms_overlap': len(forms_with_both_pos),
                            'tf_overlap_details': str(forms_with_both_pos[:5]),
                            'sample_words_pos1': ', '.join(sample_words1),
                            'sample_words_pos2': ', '.join(sample_words2),
                            'sample_context1': instances1[0].context[:100] if instances1 else '',
                            'sample_context2': instances2[0].context[:100] if instances2 else ''
                        })

                    except Exception as e:
                        logger.warning(f"Error embedding lemma '{lemma}': {e}")

                    if len(results) % 100 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        return pd.DataFrame(results)

    def generate_reports(self, embedding_results: pd.DataFrame,
                        tf_errors: List[Dict],
                        tf_polysemy: List[FormStatistics],
                        lemma_conflicts: List[Dict] = None):
        """Generate comprehensive validation reports"""
        logger.info("Generating reports...")

        # 0. Lemma-level POS conflicts (NEW - most important!)
        if lemma_conflicts:
            lemma_conflicts_df = pd.DataFrame(lemma_conflicts)
            lemma_conflicts_file = self.output_dir / 'tf_lemma_pos_conflicts.csv'
            lemma_conflicts_df.to_csv(lemma_conflicts_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Lemma POS conflicts: {lemma_conflicts_file} ({len(lemma_conflicts_df)} cases)")

            # Separate by status
            errors_df = lemma_conflicts_df[lemma_conflicts_df['status'] == 'LIKELY_ERROR']
            polysemy_df = lemma_conflicts_df[lemma_conflicts_df['status'] == 'LIKELY_POLYSEMY']
            uncertain_df = lemma_conflicts_df[lemma_conflicts_df['status'] == 'UNCERTAIN']

            if not errors_df.empty:
                errors_file = self.output_dir / 'tf_lemma_likely_errors.csv'
                errors_df.to_csv(errors_file, index=False, encoding='utf-8-sig')
                logger.info(f"✓ Lemma likely errors: {errors_file} ({len(errors_df)} cases)")

            if not polysemy_df.empty:
                polysemy_file = self.output_dir / 'tf_lemma_likely_polysemy.csv'
                polysemy_df.to_csv(polysemy_file, index=False, encoding='utf-8-sig')
                logger.info(f"✓ Lemma likely polysemy: {polysemy_file} ({len(polysemy_df)} cases)")

        # 1. TF-based form-level errors report
        tf_errors_df = pd.DataFrame(tf_errors)
        if not tf_errors_df.empty:
            tf_errors_file = self.output_dir / 'tf_form_potential_errors.csv'
            tf_errors_df.to_csv(tf_errors_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ TF form errors: {tf_errors_file} ({len(tf_errors_df)} cases)")

        # 2. TF-based polysemy report
        polysemy_data = [{
            'lemma': s.lemma,
            'form': s.form,
            'total_count': s.total_count,
            'pos_distribution': str(s.pos_distribution),
            'entropy': round(s.pos_entropy, 4),
            'consistency': round(s.consistency_score, 4)
        } for s in tf_polysemy]
        tf_polysemy_df = pd.DataFrame(polysemy_data)
        if not tf_polysemy_df.empty:
            tf_polysemy_file = self.output_dir / 'tf_genuine_polysemy.csv'
            tf_polysemy_df.to_csv(tf_polysemy_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ TF genuine polysemy: {tf_polysemy_file} ({len(tf_polysemy_df)} cases)")

        # 3. Lemma summary with TF stats
        lemma_summary = self.tf_validator.get_lemma_summary()
        lemma_summary_file = self.output_dir / 'lemma_tf_summary.csv'
        lemma_summary.to_csv(lemma_summary_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Lemma TF summary: {lemma_summary_file}")

        # 4. Embedding validation results
        if not embedding_results.empty:
            full_report = self.output_dir / 'validation_results.csv'
            embedding_results.to_csv(full_report, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Full embedding results: {full_report}")

            # Suspected errors
            errors_df = embedding_results[embedding_results['status'] == 'LIKELY_ERROR'].sort_values(
                'avg_similarity', ascending=False
            )
            errors_file = self.output_dir / 'suspected_errors.csv'
            errors_df.to_csv(errors_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Embedding suspected errors: {errors_file} ({len(errors_df)} cases)")

            # Confirmed polysemy
            polysemy_df = embedding_results[embedding_results['status'] == 'GENUINE_POLYSEMY'].sort_values(
                'avg_similarity', ascending=True
            )
            polysemy_file = self.output_dir / 'confirmed_polysemy.csv'
            polysemy_df.to_csv(polysemy_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Embedding confirmed polysemy: {polysemy_file} ({len(polysemy_df)} cases)")

            # Uncertain cases
            uncertain_df = embedding_results[embedding_results['status'] == 'UNCERTAIN'].sort_values(
                'avg_similarity', ascending=False
            )
            uncertain_file = self.output_dir / 'uncertain_cases.csv'
            uncertain_df.to_csv(uncertain_file, index=False, encoding='utf-8-sig')
            logger.info(f"✓ Uncertain cases: {uncertain_file} ({len(uncertain_df)} cases)")

        # 5. Summary JSON
        summary = {
            **self.stats,
            'thresholds': {
                'high_similarity_threshold': self.HIGH_SIMILARITY_THRESHOLD,
                'low_similarity_threshold': self.LOW_SIMILARITY_THRESHOLD,
                'tf_error_threshold': self.TF_ERROR_THRESHOLD,
                'tf_polysemy_min_entropy': self.TF_POLYSEMY_MIN_ENTROPY
            }
        }

        summary_file = self.output_dir / 'validation_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Summary: {summary_file}")

        self._print_summary(embedding_results, tf_errors, tf_polysemy, lemma_conflicts)

    def _print_summary(self, embedding_results: pd.DataFrame,
                      tf_errors: List[Dict],
                      tf_polysemy: List[FormStatistics],
                      lemma_conflicts: List[Dict] = None):
        """Print summary to console"""
        print("\n" + "="*80)
        print("LEMMATIZATION VALIDATION REPORT (TF + Embeddings)")
        print("="*80)

        print(f"\n📊 Overall Statistics:")
        print(f"   Files analyzed:         {self.stats['total_files']:,}")
        print(f"   Total words:            {self.stats['total_words']:,}")
        print(f"   Unique lemmas:          {self.stats['unique_lemmas']:,}")
        print(f"   Unique word forms:      {self.stats['unique_forms']:,}")
        print(f"   Lemmas with conflicts:  {self.stats['lemmas_with_pos_conflicts']:,}")

        print(f"\n📈 TF-Based Analysis:")
        print(f"   Form-level errors:        {self.stats['tf_potential_errors']:,}")
        print(f"   Form-level polysemy:      {self.stats['tf_genuine_polysemy']:,}")
        if 'tf_lemma_conflicts' in self.stats:
            print(f"   Lemma-level POS conflicts: {self.stats['tf_lemma_conflicts']:,}")

        if len(embedding_results) > 0:
            print(f"\n📈 Embedding Analysis:")
            print(f"   Suspected errors:       {self.stats['embedding_suspected_errors']:,}")
            print(f"   Confirmed polysemy:     {self.stats['embedding_confirmed_polysemy']:,}")
            print(f"   Uncertain cases:        {self.stats['uncertain_cases']:,}")

        # Show lemma-level conflicts first (most important!)
        if lemma_conflicts:
            likely_errors = [c for c in lemma_conflicts if c['status'] == 'LIKELY_ERROR']
            likely_polysemy = [c for c in lemma_conflicts if c['status'] == 'LIKELY_POLYSEMY']

            print(f"\n🔝 Top 15 Lemma-Level POS Conflicts:")
            print("-"*100)
            print(f"{'Lemma':<15}{'Dom.POS':<12}{'Count':>6}{'Min.POS':<12}{'Count':>6}{'Ratio':>8}{'Status':<15}{'Sample Forms'}")
            print("-"*100)
            for c in lemma_conflicts[:15]:
                print(f"{c['lemma']:<15}{c['dominant_pos']:<12}{c['dominant_count']:>6}{c['minority_pos']:<12}{c['minority_count']:>6}{c['minority_ratio']:>8.2%}  {c['status']:<15}{c['sample_minority_forms'][:30]}")

        if tf_errors:
            print(f"\n🔝 Top 10 Form-Level Potential Errors:")
            print("-"*80)
            print(f"{'Form':<20}{'Lemma':<15}{'Dominant':<12}{'Count':>6}{'Minority':<12}{'Count':>6}{'Conf':>6}")
            print("-"*80)
            for err in tf_errors[:10]:
                print(f"{err['form']:<20}{err['lemma']:<15}{err['dominant_pos']:<12}{err['dominant_count']:>6}{err['minority_pos']:<12}{err['minority_count']:>6}{err['confidence']:>6.2f}")

        if tf_polysemy:
            print(f"\n🔍 Top 10 Genuine Polysemy (by entropy):")
            print("-"*80)
            print(f"{'Form':<20}{'Lemma':<15}{'Count':>8}{'Entropy':>8}{'POS Distribution'}")
            print("-"*80)
            for stats in tf_polysemy[:10]:
                pos_str = ', '.join([f"{p}:{c}" for p, c in stats.pos_distribution.items()])
                print(f"{stats.form:<20}{stats.lemma:<15}{stats.total_count:>8}{stats.pos_entropy:>8.3f}  {pos_str[:40]}")

        print("\n" + "="*80)
        print(f"✅ Reports saved to: {self.output_dir}/")
        print("="*80)

    def run(self, file_limit: int = None, max_conflicts: int = 500, use_embeddings: bool = True):
        """
        Run the full validation pipeline.

        Args:
            file_limit: Max files to load
            max_conflicts: Max conflicts to validate with embeddings
            use_embeddings: Whether to use embedding validation (slower but more accurate)
        """
        # Step 1: Load lemmatized files
        self.load_lemmatized_files(limit=file_limit)

        # Step 2: Run TF analysis (now returns 3 values)
        tf_errors, tf_polysemy, lemma_conflicts = self.run_tf_analysis()

        # Step 3: Find POS conflicts
        conflicts = self.find_pos_conflicts()

        embedding_results = pd.DataFrame()

        if use_embeddings and conflicts:
            # Limit conflicts for processing
            if len(conflicts) > max_conflicts:
                logger.info(f"Limiting to top {max_conflicts} conflicts by frequency")
                sorted_conflicts = sorted(
                    conflicts.items(),
                    key=lambda x: sum(len(v) for v in x[1].values()),
                    reverse=True
                )[:max_conflicts]
                conflicts = dict(sorted_conflicts)

            # Step 4: Validate with embeddings
            embedding_results = self.validate_with_embeddings(conflicts)

        # Step 5: Generate reports
        self.generate_reports(embedding_results, tf_errors, tf_polysemy, lemma_conflicts)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate lemmatization using TF analysis and AraBERT embeddings')
    parser.add_argument('--input', default=None, help='Directory with lemmatized JSON files')
    parser.add_argument('--output', default=None, help='Output directory for reports')
    parser.add_argument('--file-limit', type=int, help='Limit number of files to process')
    parser.add_argument('--max-conflicts', type=int, default=500, help='Max conflicts to validate with embeddings')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embedding validation (faster)')

    args = parser.parse_args()

    validator = LemmatizationValidator(
        lemmatized_dir=args.input,
        output_dir=args.output
    )

    validator.run(
        file_limit=args.file_limit,
        max_conflicts=args.max_conflicts,
        use_embeddings=not args.no_embeddings
    )


if __name__ == "__main__":
    main()

