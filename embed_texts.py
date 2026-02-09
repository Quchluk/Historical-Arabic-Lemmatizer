#!/usr/bin/env python3
"""
Text Embeddings Generator for OpenITI Corpus
Embeds all text files at token level using AraBERT v2
Excludes metadata headers from embedding
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModel
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_embedding.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TextEmbedding:
    """Container for text embedding data"""
    version_uri: str
    file_path: str
    text_content: str
    text_length_chars: int
    text_length_words: int
    word_embeddings: np.ndarray  # Shape: (num_words, embedding_dim) - mean pooled
    words: List[str]  # List of words
    pooled_embedding: np.ndarray  # Shape: (embedding_dim,) - mean of all word embeddings


class TextExtractor:
    """Extract clean text from OpenITI files, excluding metadata"""

    @staticmethod
    def extract_text_content(file_path: str) -> Tuple[str, int]:
        """
        Extract text content from OpenITI file, excluding metadata header

        Returns:
            Tuple of (text_content, start_line_number)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Find where metadata ends
            metadata_end_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#META#Header#End#'):
                    metadata_end_idx = i + 1
                    break

            # If no metadata header found, look for ######OpenITI#
            if metadata_end_idx == 0:
                for i, line in enumerate(lines):
                    if line.strip().startswith('######OpenITI#'):
                        # Skip until we find actual content (non-META lines)
                        for j in range(i + 1, len(lines)):
                            if not lines[j].strip().startswith('#META#') and lines[j].strip():
                                metadata_end_idx = j
                                break
                        break

            # Extract text after metadata
            text_lines = lines[metadata_end_idx:]

            # Clean the text - remove page markers and other markup
            clean_lines = []
            for line in text_lines:
                line = line.strip()

                # Skip empty lines and special markers
                if not line:
                    continue
                if line.startswith('PageV'):
                    continue
                if line.startswith('###'):
                    continue
                if line.startswith('~~'):
                    line = line[2:].strip()
                if line.startswith('#'):
                    line = line[1:].strip()

                if line:
                    clean_lines.append(line)

            text_content = ' '.join(clean_lines)

            return text_content, metadata_end_idx

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return "", 0


class AraBERTEmbedder:
    """Generate embeddings using AraBERT v2"""

    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2", device: str = None):
        """
        Initialize AraBERT embedder

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'mps', or 'cpu'. If None, auto-detect
        """
        logger.info(f"Initializing AraBERT model: {model_name}")

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def reconstruct_words_from_tokens(self, tokens: List[str], token_embeddings: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Reconstruct words from subword tokens and mean-pool their embeddings

        Args:
            tokens: List of tokens (including subwords with ##)
            token_embeddings: Token-level embeddings (num_tokens, embedding_dim)

        Returns:
            Tuple of (words, word_embeddings)
        """
        words = []
        word_embeddings = []

        idx = 0
        while idx < len(tokens):
            token = tokens[idx]

            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                idx += 1
                continue

            # Start of a new word
            word_parts = [token]
            word_token_embeddings = [token_embeddings[idx]]
            idx += 1

            # Collect subword tokens (those starting with ##)
            while idx < len(tokens) and tokens[idx].startswith('##'):
                word_parts.append(tokens[idx][2:])  # Remove ##
                word_token_embeddings.append(token_embeddings[idx])
                idx += 1

            # Reconstruct word
            word = ''.join(word_parts)

            # Mean pool embeddings for this word
            word_embedding = np.mean(word_token_embeddings, axis=0)

            words.append(word)
            word_embeddings.append(word_embedding)

        return words, np.array(word_embeddings)

    def embed_text(self, text: str, max_length: int = 512,
                   batch_size: int = 8) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Generate word-level embeddings for text (mean-pooled from tokens)

        Args:
            text: Input text to embed
            max_length: Maximum sequence length for model
            batch_size: Number of chunks to process at once

        Returns:
            Tuple of (word_embeddings, words, pooled_embedding)
        """
        if not text or not text.strip():
            # Return empty embeddings for empty text
            return (np.zeros((0, self.embedding_dim)),
                   [],
                   np.zeros(self.embedding_dim))

        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)

        # If text is too long, we need to process in chunks
        if len(tokens) > max_length - 2:  # -2 for [CLS] and [SEP]
            return self._embed_long_text(text, tokens, max_length, batch_size)
        else:
            return self._embed_short_text(text, tokens)

    def _embed_short_text(self, text: str, tokens: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Embed text that fits in one sequence and return word-level embeddings"""
        # Encode the text
        encoded = self.tokenizer(text,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=512)

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               output_hidden_states=True)

        # Get token embeddings (last hidden state)
        token_embeddings = outputs.last_hidden_state[0].cpu().numpy()  # Shape: (seq_len, hidden_dim)

        # Convert token IDs back to tokens
        token_ids = input_ids[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # Reconstruct words and mean-pool their embeddings
        words, word_embeddings = self.reconstruct_words_from_tokens(tokens, token_embeddings)

        # Compute pooled embedding as mean of all word embeddings
        if len(word_embeddings) > 0:
            pooled_embedding = np.mean(word_embeddings, axis=0)
        else:
            pooled_embedding = np.zeros(self.embedding_dim)

        return word_embeddings, words, pooled_embedding

    def _embed_long_text(self, text: str, tokens: List[str],
                        max_length: int, batch_size: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Embed long text by processing in chunks split on word boundaries
        Returns word-level embeddings (mean-pooled from subword tokens)
        """
        # Split text into words (by spaces)
        words = text.split()

        chunk_size = max_length - 2  # Reserve space for [CLS] and [SEP]

        all_token_embeddings = []
        all_tokens = []

        # Process words in chunks
        word_idx = 0

        while word_idx < len(words):
            # Build chunk by adding words until we reach max_length
            chunk_words = []
            chunk_token_count = 0

            while word_idx < len(words) and chunk_token_count < chunk_size:
                word = words[word_idx]
                word_tokens = self.tokenizer.tokenize(word)

                # Check if adding this word would exceed chunk_size
                if chunk_token_count + len(word_tokens) > chunk_size and chunk_words:
                    break  # Don't add this word, process current chunk

                chunk_words.append(word)
                chunk_token_count += len(word_tokens)
                word_idx += 1

            if not chunk_words:
                # Single word is too long, force it into a chunk
                chunk_words = [words[word_idx]]
                word_idx += 1

            # Join words back into text
            chunk_text = ' '.join(chunk_words)

            # Encode chunk
            encoded = self.tokenizer(chunk_text,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=max_length)

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   output_hidden_states=True)

            chunk_token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            chunk_token_ids = input_ids[0].cpu().tolist()
            chunk_tokens = self.tokenizer.convert_ids_to_tokens(chunk_token_ids)

            # Skip [CLS] and [SEP] tokens except for first and last chunk
            start_offset = 1 if all_token_embeddings else 0  # Skip [CLS] if not first chunk
            end_offset = -1 if word_idx < len(words) else 0  # Skip [SEP] if not last chunk

            if start_offset > 0 or end_offset < 0:
                chunk_token_embeddings = chunk_token_embeddings[start_offset if start_offset > 0 else None:
                                                   end_offset if end_offset < 0 else None]
                chunk_tokens = chunk_tokens[start_offset if start_offset > 0 else None:
                                           end_offset if end_offset < 0 else None]

            all_token_embeddings.append(chunk_token_embeddings)
            all_tokens.extend(chunk_tokens)

        # Concatenate all token embeddings
        token_embeddings = np.vstack(all_token_embeddings) if all_token_embeddings else np.zeros((0, self.embedding_dim))

        # Reconstruct words from all tokens and mean-pool
        reconstructed_words, word_embeddings = self.reconstruct_words_from_tokens(all_tokens, token_embeddings)

        # Compute pooled embedding as mean of all word embeddings
        if len(word_embeddings) > 0:
            pooled_embedding = np.mean(word_embeddings, axis=0)
        else:
            pooled_embedding = np.zeros(self.embedding_dim)

        return word_embeddings, reconstructed_words, pooled_embedding


class TextEmbeddingGenerator:
    """Main orchestrator for generating embeddings for all texts"""

    def __init__(self, db_root: str, metadata_file: str = "openiti_metadata.parquet",
                 output_dir: str = "embeddings", batch_size: int = 1):
        """
        Initialize the embedding generator

        Args:
            db_root: Root directory of the OpenITI database
            metadata_file: Path to metadata parquet file
            output_dir: Directory to save embeddings
            batch_size: Number of texts to process before saving checkpoint
        """
        self.db_root = Path(db_root)
        self.metadata_file = metadata_file
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.text_extractor = TextExtractor()
        self.embedder = AraBERTEmbedder()

        # Load metadata
        logger.info(f"Loading metadata from {metadata_file}")
        self.metadata_df = pd.read_parquet(metadata_file)
        logger.info(f"Loaded metadata for {len(self.metadata_df):,} texts")

        # Statistics
        self.stats = {
            'total_texts': 0,
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'total_tokens': 0
        }

    def process_all_texts(self, limit: Optional[int] = None,
                         save_individual: bool = True,
                         save_combined: bool = True):
        """
        Process all texts and generate embeddings

        Args:
            limit: Maximum number of texts to process (None for all)
            save_individual: Save individual embedding files
            save_combined: Save combined embedding index
        """
        logger.info("Starting embedding generation...")

        # Filter to texts with valid file paths
        valid_texts = self.metadata_df[self.metadata_df['text_file_path'] != ''].copy()

        if limit:
            valid_texts = valid_texts.head(limit)

        self.stats['total_texts'] = len(valid_texts)
        logger.info(f"Processing {len(valid_texts):,} texts")

        # Storage for combined embeddings
        embedding_index = []

        # Process each text
        for idx, row in tqdm(valid_texts.iterrows(),
                            total=len(valid_texts),
                            desc="Generating embeddings"):
            try:
                # Extract text content (excluding metadata)
                text_content, _ = self.text_extractor.extract_text_content(row['text_file_path'])

                if not text_content or len(text_content.strip()) < 10:
                    logger.warning(f"Skipping {row['version_uri']}: insufficient content")
                    self.stats['skipped'] += 1
                    continue

                # Generate embeddings
                word_embeddings, words, pooled_embedding = self.embedder.embed_text(text_content)

                # Create embedding object
                embedding = TextEmbedding(
                    version_uri=row['version_uri'],
                    file_path=row['text_file_path'],
                    text_content=text_content[:1000],  # Store first 1000 chars as sample
                    text_length_chars=len(text_content),
                    text_length_words=len(words),
                    word_embeddings=word_embeddings,
                    words=words,
                    pooled_embedding=pooled_embedding
                )

                # Save individual embedding file
                if save_individual:
                    self._save_embedding(embedding)

                # Add to index
                embedding_index.append({
                    'version_uri': embedding.version_uri,
                    'file_path': embedding.file_path,
                    'text_length_chars': embedding.text_length_chars,
                    'text_length_words': embedding.text_length_words,
                    'embedding_file': self._get_embedding_filename(embedding.version_uri)
                })

                self.stats['processed'] += 1
                self.stats['total_tokens'] += len(words)

                # Periodic checkpoint
                if self.stats['processed'] % 100 == 0:
                    logger.info(f"Processed {self.stats['processed']:,} texts, "
                              f"{self.stats['total_tokens']:,} words total")

            except Exception as e:
                logger.error(f"Error processing {row['version_uri']}: {e}")
                self.stats['errors'] += 1
                continue

        # Save combined index
        if save_combined and embedding_index:
            self._save_embedding_index(embedding_index)

        self._log_final_statistics()

    def _save_embedding(self, embedding: TextEmbedding):
        """Save individual embedding to file"""
        filename = self._get_embedding_filename(embedding.version_uri)
        filepath = self.output_dir / filename

        with open(filepath, 'wb') as f:
            pickle.dump(embedding, f)

    def _get_embedding_filename(self, version_uri: str) -> str:
        """Generate filename for embedding"""
        # Sanitize URI for filename
        safe_name = re.sub(r'[^\w\-\.]', '_', version_uri)
        return f"{safe_name}.pkl"

    def _save_embedding_index(self, embedding_index: List[Dict]):
        """Save index of all embeddings"""
        index_df = pd.DataFrame(embedding_index)

        # Save as parquet
        index_file = self.output_dir / "embedding_index.parquet"
        index_df.to_parquet(index_file, index=False)
        logger.info(f"Saved embedding index to {index_file}")

        # Also save as CSV for easy inspection
        csv_file = self.output_dir / "embedding_index.csv"
        index_df.to_csv(csv_file, index=False)
        logger.info(f"Saved embedding index to {csv_file}")

    def _log_final_statistics(self):
        """Log final statistics"""
        logger.info("=" * 80)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total texts:     {self.stats['total_texts']:,}")
        logger.info(f"Processed:       {self.stats['processed']:,}")
        logger.info(f"Skipped:         {self.stats['skipped']:,}")
        logger.info(f"Errors:          {self.stats['errors']:,}")
        logger.info(f"Total words:     {self.stats['total_tokens']:,}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate token-level embeddings for OpenITI texts using AraBERT')
    parser.add_argument('--db-root', default='db', help='Root directory of OpenITI database')
    parser.add_argument('--metadata', default='openiti_metadata.parquet', help='Metadata parquet file')
    parser.add_argument('--output-dir', default='embeddings', help='Output directory for embeddings')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of texts to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--no-individual', action='store_true', help='Do not save individual embedding files')
    parser.add_argument('--no-combined', action='store_true', help='Do not save combined index')

    args = parser.parse_args()

    # Create generator
    generator = TextEmbeddingGenerator(
        db_root=args.db_root,
        metadata_file=args.metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    # Process all texts
    generator.process_all_texts(
        limit=args.limit,
        save_individual=not args.no_individual,
        save_combined=not args.no_combined
    )

    logger.info("✓ All done!")


if __name__ == '__main__':
    main()

