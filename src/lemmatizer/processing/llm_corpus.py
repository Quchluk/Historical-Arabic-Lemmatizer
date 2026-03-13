#!/usr/bin/env python3
"""
Arabic Corpus Lemmatizer using LLM (DeepSeek via OpenRouter)
Processes all text files from db/ directory and outputs lemmatized JSON files
with structure: {source_file, processed_at, num_words, words_and_lemmas: [{word, lemma, pos}]}
"""
import json
import os
import re
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Use faster model - deepseek-chat is much faster than deepseek-r1 for this task
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Directories
DB_DIR = Path("db")
OUTPUT_DIR = Path("llm_lemmatized")
CHECKPOINT_FILE = Path("llm_lemmatize_checkpoint.json")
TF_CACHE_FILE = Path("tf_lemma_cache.json")

# Processing settings
BATCH_SIZE = 200  # Words per API call (LLMs can handle large batches)
API_TIMEOUT = 300  # Seconds (5 min for large batches)
RETRY_DELAY = 2  # Seconds
MAX_RETRIES = 3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_lemmatize_corpus.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TFLemmaCache:
    """
    TF-based lemma cache that learns from API responses.
    Uses term frequency to resolve ambiguous cases.
    """

    def __init__(self, cache_file: Path = TF_CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict] = {}  # word -> {lemma, pos, count}
        self.tf_matrix: Dict[str, Counter] = {}  # word -> Counter({(lemma, pos): count})
        self.load_cache()

    def load_cache(self):
        """Load existing cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    # Convert tf_matrix back to Counter objects
                    tf_raw = data.get('tf_matrix', {})
                    for word, counts in tf_raw.items():
                        self.tf_matrix[word] = Counter()
                        for key_str, count in counts.items():
                            # Parse "(lemma, pos)" string back to tuple
                            self.tf_matrix[word][key_str] = count
                logger.info(f"Loaded cache with {len(self.cache)} words")
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")

    def save_cache(self):
        """Save cache to file."""
        try:
            # Convert Counter objects to regular dicts for JSON
            tf_serializable = {}
            for word, counter in self.tf_matrix.items():
                tf_serializable[word] = dict(counter)

            data = {
                'cache': self.cache,
                'tf_matrix': tf_serializable,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache: {e}")

    def get(self, word: str) -> Optional[Dict]:
        """Get cached lemma for word. Returns most frequent (lemma, pos) pair."""
        if word in self.tf_matrix:
            # Return the most common (lemma, pos) pair
            most_common = self.tf_matrix[word].most_common(1)
            if most_common:
                key_str = most_common[0][0]
                # Parse "lemma|pos" format
                parts = key_str.split('|')
                if len(parts) == 2:
                    return {'lemma': parts[0], 'pos': parts[1]}
        return self.cache.get(word)

    def update(self, word: str, lemma: str, pos: str):
        """Update TF matrix with new observation."""
        if word not in self.tf_matrix:
            self.tf_matrix[word] = Counter()

        key = f"{lemma}|{pos}"
        self.tf_matrix[word][key] += 1

        # Update simple cache with most frequent
        most_common = self.tf_matrix[word].most_common(1)
        if most_common:
            key_str = most_common[0][0]
            parts = key_str.split('|')
            if len(parts) == 2:
                self.cache[word] = {'lemma': parts[0], 'pos': parts[1]}

    def batch_update(self, results: List[Dict]):
        """Update cache with batch of results."""
        for item in results:
            word = item.get('word', '')
            lemma = item.get('lemma', '')
            pos = item.get('pos', '')
            if word and lemma:
                self.update(word, lemma, pos)


class LLMLemmatizer:
    """Lemmatizer using OpenRouter API with DeepSeek model."""

    def __init__(self, cache: TFLemmaCache):
        self.cache = cache
        self.session = requests.Session()

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set in .env file")

        logger.info(f"Using model: {OPENROUTER_MODEL}")

    def lemmatize_batch(self, words: List[str]) -> List[Dict]:
        """Send batch of words to LLM for lemmatization."""

        # Check cache first
        cached_results = []
        uncached_words = []
        uncached_indices = []

        for i, word in enumerate(words):
            cached = self.cache.get(word)
            if cached:
                cached_results.append({
                    'word': word,
                    'lemma': cached['lemma'],
                    'pos': cached['pos'],
                    'cached': True
                })
            else:
                uncached_words.append(word)
                uncached_indices.append(i)
                cached_results.append(None)  # Placeholder

        if not uncached_words:
            return [r for r in cached_results if r is not None]

        # Call API for uncached words
        prompt = f"""أنت محلل صرفي للغة العربية. لكل كلمة عربية، قدم الجذع (lemma) وتصنيف الكلمة (POS).

الكلمات: {json.dumps(uncached_words, ensure_ascii=False)}

أعد فقط مصفوفة JSON صالحة حيث كل عنصر يحتوي على:
{{"word": "الكلمة الأصلية", "lemma": "الجذع", "pos": "التصنيف"}}

تصنيفات POS المتاحة:
- noun: اسم
- noun_prop: اسم علم
- verb: فعل
- adj: صفة
- adv: ظرف
- pron: ضمير
- prep: حرف جر
- conj: حرف عطف
- part: حرف/أداة
- num: عدد
- interj: تعجب
- punct: علامة ترقيم

لا تضف شروحات، فقط مصفوفة JSON."""

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=API_TIMEOUT
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # Handle thinking tags from DeepSeek R1
                if '<think>' in content:
                    # Extract content after </think>
                    think_end = content.find('</think>')
                    if think_end != -1:
                        content = content[think_end + 8:].strip()

                # Extract JSON from response
                content = self._extract_json(content)

                # Parse JSON
                api_results = json.loads(content)

                # Merge API results back
                api_idx = 0
                final_results = []
                for i, cached in enumerate(cached_results):
                    if cached is not None:
                        final_results.append(cached)
                    else:
                        if api_idx < len(api_results):
                            item = api_results[api_idx]
                            final_results.append({
                                'word': item.get('word', words[i]),
                                'lemma': item.get('lemma', words[i]),
                                'pos': item.get('pos', 'noun'),
                                'cached': False
                            })
                        else:
                            final_results.append({
                                'word': words[i],
                                'lemma': words[i],
                                'pos': 'noun',
                                'cached': False
                            })
                        api_idx += 1

                # Update cache with new results
                new_results = [r for r in final_results if not r.get('cached', True)]
                self.cache.batch_update(new_results)

                return final_results

            except requests.exceptions.RequestException as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                logger.debug(f"Content: {content[:500]}...")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        # Fallback: return words as-is
        return [{'word': w, 'lemma': w, 'pos': 'noun', 'cached': False} for w in words]

    def _extract_json(self, content: str) -> str:
        """Extract JSON array from response content."""
        # Remove markdown code blocks
        if '```' in content:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if match:
                content = match.group(1).strip()

        # Find JSON array
        start = content.find('[')
        end = content.rfind(']')
        if start != -1 and end != -1:
            content = content[start:end + 1]

        return content


class CorpusProcessor:
    """Process entire corpus from db/ directory."""

    def __init__(self, lemmatizer: LLMLemmatizer, cache: TFLemmaCache):
        self.lemmatizer = lemmatizer
        self.cache = cache
        self.checkpoint = self._load_checkpoint()

        OUTPUT_DIR.mkdir(exist_ok=True)

    def _load_checkpoint(self) -> Dict:
        """Load processing checkpoint."""
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'processed_files': [], 'stats': {'files': 0, 'words': 0}}

    def _save_checkpoint(self):
        """Save processing checkpoint."""
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint, f, ensure_ascii=False, indent=2)

    def find_text_files(self) -> List[Path]:
        """Find all text files in db/ directory."""
        text_files = []

        for root, dirs, files in os.walk(DB_DIR):
            for file in files:
                if file.endswith(('-ara1', '-ara2', '.txt')) and not file.startswith('.'):
                    text_files.append(Path(root) / file)

        logger.info(f"Found {len(text_files)} text files in {DB_DIR}")
        return text_files

    def extract_words(self, text: str) -> List[str]:
        """Extract Arabic words from text."""
        # Arabic Unicode range
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        words = arabic_pattern.findall(text)
        return words

    def process_file(self, file_path: Path) -> Optional[Dict]:
        """Process a single text file."""
        try:
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()

            # Extract words
            words = self.extract_words(text)

            if not words:
                logger.warning(f"No Arabic words found in {file_path}")
                return None

            logger.info(f"Processing {file_path.name}: {len(words)} words in {(len(words) + BATCH_SIZE - 1) // BATCH_SIZE} batches")

            # Process in batches
            all_results = []
            total_batches = (len(words) + BATCH_SIZE - 1) // BATCH_SIZE
            for batch_num, i in enumerate(range(0, len(words), BATCH_SIZE), 1):
                batch = words[i:i + BATCH_SIZE]
                logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} words...")
                batch_results = self.lemmatizer.lemmatize_batch(batch)
                all_results.extend(batch_results)
                cached_count = sum(1 for r in batch_results if r.get('cached', False))
                logger.info(f"  Batch {batch_num}/{total_batches}: ✓ Done ({cached_count} cached)")

                # Rate limiting
                if i + BATCH_SIZE < len(words):
                    time.sleep(0.5)

            # Build output structure
            output = {
                "source_file": str(file_path),
                "processed_at": datetime.now().isoformat(),
                "num_words": len(words),
                "words_and_lemmas": [
                    {
                        "word": r['word'],
                        "lemma": r['lemma'],
                        "pos": r['pos']
                    }
                    for r in all_results
                ]
            }

            return output

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def save_output(self, output: Dict, source_path: Path) -> Path:
        """Save lemmatized output to JSON file."""
        # Create output filename matching source structure
        relative_path = source_path.relative_to(DB_DIR)
        output_name = source_path.stem + "_lemmatized.json"
        output_path = OUTPUT_DIR / output_name

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return output_path

    def run(self, limit: Optional[int] = None, workers: int = 1):
        """Process all files in corpus."""
        logger.info("=" * 80)
        logger.info("LLM CORPUS LEMMATIZER")
        logger.info("=" * 80)
        logger.info(f"Model: {OPENROUTER_MODEL}")
        logger.info(f"Input: {DB_DIR}")
        logger.info(f"Output: {OUTPUT_DIR}")
        logger.info(f"Batch size: {BATCH_SIZE}")

        # Find files
        all_files = self.find_text_files()

        # Filter already processed
        processed_set = set(self.checkpoint.get('processed_files', []))
        files_to_process = [f for f in all_files if str(f) not in processed_set]

        if processed_set:
            logger.info(f"✓ Resuming: {len(processed_set)} files already processed")

        if limit:
            files_to_process = files_to_process[:limit]

        logger.info(f"Files to process: {len(files_to_process)}")
        logger.info("=" * 80)

        if not files_to_process:
            logger.info("No files to process!")
            return

        # Process files
        stats = self.checkpoint.get('stats', {'files': 0, 'words': 0})

        for file_path in tqdm(files_to_process, desc="Lemmatizing"):
            try:
                output = self.process_file(file_path)

                if output:
                    # Save output
                    self.save_output(output, file_path)

                    # Update stats
                    stats['files'] += 1
                    stats['words'] += output['num_words']

                    # Update checkpoint
                    self.checkpoint['processed_files'].append(str(file_path))
                    self.checkpoint['stats'] = stats

                    # Save periodically
                    if stats['files'] % 10 == 0:
                        self._save_checkpoint()
                        self.cache.save_cache()

            except KeyboardInterrupt:
                logger.info("\n⚠️ Interrupted! Saving progress...")
                self._save_checkpoint()
                self.cache.save_cache()
                raise
            except Exception as e:
                logger.error(f"Error with {file_path}: {e}")
                continue

        # Final save
        self._save_checkpoint()
        self.cache.save_cache()

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Files processed: {stats['files']}")
        logger.info(f"  Total words: {stats['words']:,}")
        logger.info(f"  Cache size: {len(self.cache.cache):,} words")
        logger.info(f"  Output directory: {OUTPUT_DIR}")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='LLM Corpus Lemmatizer')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=50, help='Words per API call')
    parser.add_argument('--reset', action='store_true', help='Reset checkpoint and start fresh')
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    # Reset if requested
    if args.reset:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            logger.info("Checkpoint reset")

    # Initialize components
    cache = TFLemmaCache()
    lemmatizer = LLMLemmatizer(cache)
    processor = CorpusProcessor(lemmatizer, cache)

    # Run
    try:
        processor.run(limit=args.limit, workers=args.workers)
    except KeyboardInterrupt:
        logger.info("Stopped by user")


if __name__ == "__main__":
    main()

