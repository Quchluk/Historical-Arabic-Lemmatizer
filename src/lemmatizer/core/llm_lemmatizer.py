#!/usr/bin/env python3
"""
Arabic Word Lemmatizer with OpenRouter API
Outputs: CSV with columns: word, frequency, lemma
Uses DeepSeek model via OpenRouter
"""
import csv
import json
import time
import os
import requests
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration from .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# File paths
COMBINED_WORD_FREQUENCIES = Path("word_frequencies.json")
LEMMATIZED_WORDS_CSV = Path("lemmatized_words.csv")
LEMMATIZE_CHECKPOINT = Path("lemmatize_checkpoint.json")

# Processing settings
BATCH_SIZE = 20  # Words per API call
API_TIMEOUT = 120  # Seconds
RETRY_DELAY = 2  # Seconds between batches


def load_words() -> List[Dict[str, any]]:
    """Load word frequencies from input file."""
    print(f"Loading words from {COMBINED_WORD_FREQUENCIES}...")
    with open(COMBINED_WORD_FREQUENCIES, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_freqs = data.get("word_frequencies", {})
    words = [{"word": word, "frequency": freq} for word, freq in word_freqs.items()]
    words.sort(key=lambda x: x["frequency"], reverse=True)

    print(f"Loaded {len(words)} unique words")
    return words


def load_checkpoint() -> Dict:
    """Load checkpoint if exists."""
    if LEMMATIZE_CHECKPOINT.exists():
        with open(LEMMATIZE_CHECKPOINT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": 0, "results": []}


def save_checkpoint(processed: int, results: List[Dict]):
    """Save checkpoint."""
    with open(LEMMATIZE_CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump({"processed": processed, "results": results}, f, ensure_ascii=False, indent=2)


def lemmatize_batch(words: List[Dict], session: requests.Session) -> List[Dict]:
    """Send batch to OpenRouter for lemmatization."""

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file.")

    # Create a simple word list for the prompt
    word_list = [w["word"] for w in words]

    prompt = f"""You are an Arabic lemmatizer. For each Arabic word below, provide its lemma (root/base form).

Words: {json.dumps(word_list, ensure_ascii=False)}

Return ONLY a valid JSON array where each element has: {{"word": "original", "lemma": "base_form"}}
No explanations, no markdown, just the JSON array."""

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = session.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()

    # Extract JSON from response (handle markdown code blocks)
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

    # Parse JSON
    lemmas = json.loads(content)

    # Merge with frequencies
    output = []
    for i, word_data in enumerate(words):
        lemma_info = lemmas[i] if i < len(lemmas) else {}
        output.append({
            "word": word_data["word"],
            "frequency": word_data["frequency"],
            "lemma": lemma_info.get("lemma", word_data["word"])
        })

    return output


def save_to_csv(results: List[Dict], filepath):
    """Save results to CSV file."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "frequency", "lemma"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} rows to {filepath}")


def main():
    print("Starting Arabic Lemmatization\n")

    # Load input
    all_words = load_words()
    total = len(all_words)

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed = checkpoint["processed"]
    results = checkpoint["results"]

    if processed > 0:
        print(f"Resuming from word {processed}/{total}\n")

    # Create session
    session = requests.Session()

    # Process in batches
    current = processed
    while current < total:
        batch_end = min(current + BATCH_SIZE, total)
        batch = all_words[current:batch_end]

        print(f"Processing words {current+1}-{batch_end}/{total}...")

        try:
            batch_results = lemmatize_batch(batch, session)
            results.extend(batch_results)
            current = batch_end

            # Save checkpoint every batch
            save_checkpoint(current, results)
            print(f"   Batch complete ({len(results)} total)\n")

            # Rate limiting
            if current < total:
                time.sleep(RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"   Network error: {e}")
            print(f"   Retrying in {RETRY_DELAY * 2} seconds...\n")
            time.sleep(RETRY_DELAY * 2)

        except json.JSONDecodeError as e:
            print(f"   JSON parsing error: {e}")
            print(f"   Skipping batch and continuing...\n")
            current = batch_end  # Skip this batch

        except Exception as e:
            print(f"   Error: {e}")
            print(f"   Saving progress and stopping...\n")
            save_checkpoint(current, results)
            return

    # Save final CSV
    print("\nAll words processed!")
    save_to_csv(results, LEMMATIZED_WORDS_CSV)

    # Clean up checkpoint
    if LEMMATIZE_CHECKPOINT.exists():
        LEMMATIZE_CHECKPOINT.unlink()
        print("Checkpoint cleaned up")

    print("\nDone!")


if __name__ == "__main__":
    main()
