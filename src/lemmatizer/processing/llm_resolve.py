#!/usr/bin/env python3
"""
LLM-Based Lemmatization Resolver
=================================
Uses OpenRouter API with Gemini to analyze uncertain POS tagging cases
and determine if they are errors or genuine polysemy.
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time
from dataclasses import dataclass
import requests
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use os.environ directly

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_resolution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ResolutionCase:
    """A case to be resolved by the LLM"""
    lemma: str
    pos1: str
    pos2: str
    sample_words_pos1: str
    sample_words_pos2: str
    sample_context1: str
    sample_context2: str
    avg_similarity: float
    status: str


class OpenRouterClient:
    """Client for OpenRouter API"""

    def __init__(self, api_key: str, model: str = "deepseek/deepseek-r1"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/lemmatizer",  # Optional
            "X-Title": "Arabic Lemmatization Validator"  # Optional
        }

    def query(self, prompt: str, system_prompt: str = None, temperature: float = 0.1) -> str:
        """
        Send a query to the LLM and get response.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature (low = more deterministic)

        Returns:
            LLM response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse response: {e}")
            return None


class LemmaResolver:
    """Resolve ambiguous lemmatization cases using LLM"""

    SYSTEM_PROMPT = """أنت خبير في اللغة العربية والصرف والنحو. مهمتك هي تحليل حالات الإعراب الملتبسة في النصوص العربية.

You are an expert in Arabic linguistics, morphology, and syntax. Your task is to analyze ambiguous POS tagging cases.

For each case, you will be given:
1. A lemma (root/base form)
2. Two different POS tags assigned to words with this lemma
3. Sample words and contexts for each POS

You must determine:
1. Is this a GENUINE case of polysemy (the lemma truly has multiple POS based on meaning)?
2. Or is this a TAGGING ERROR (the words have the same meaning but were incorrectly tagged)?

Respond in JSON format only:
{
    "decision": "POLYSEMY" or "ERROR" or "UNCERTAIN",
    "confidence": 0.0-1.0,
    "correct_pos": "the correct POS if ERROR, null otherwise",
    "explanation_ar": "تفسير بالعربية",
    "explanation_en": "Explanation in English"
}"""

    def __init__(self, api_key: str, model: str = "deepseek/deepseek-r1"):
        self.client = OpenRouterClient(api_key, model)
        self.results = []

    def build_prompt(self, case: ResolutionCase) -> str:
        """Build the analysis prompt for a case"""
        prompt = f"""تحليل حالة إعراب ملتبسة:
Analyze this ambiguous POS tagging case:

## اللمة (Lemma): {case.lemma}

### الحالة الأولى (Case 1):
- **POS Tag**: {case.pos1}
- **Sample Words**: {case.sample_words_pos1}
- **Context**: {case.sample_context1}

### الحالة الثانية (Case 2):
- **POS Tag**: {case.pos2}
- **Sample Words**: {case.sample_words_pos2}
- **Context**: {case.sample_context2}

### معلومات إضافية (Additional Info):
- Embedding Similarity: {case.avg_similarity:.3f} (higher = more similar meanings)
- Current Status: {case.status}

هل هذه حالة تعدد معاني حقيقية (polysemy) أم خطأ في التصنيف (tagging error)؟
Is this genuine polysemy or a tagging error?

Respond in JSON format only."""

        return prompt

    def resolve_case(self, case: ResolutionCase) -> Dict:
        """Resolve a single case using LLM"""
        prompt = self.build_prompt(case)

        response = self.client.query(prompt, self.SYSTEM_PROMPT)

        if not response:
            return {
                "lemma": case.lemma,
                "pos1": case.pos1,
                "pos2": case.pos2,
                "decision": "API_ERROR",
                "confidence": 0.0,
                "correct_pos": None,
                "explanation_ar": "فشل في الاتصال",
                "explanation_en": "API connection failed"
            }

        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            return {
                "lemma": case.lemma,
                "pos1": case.pos1,
                "pos2": case.pos2,
                "original_status": case.status,
                "avg_similarity": case.avg_similarity,
                "sample_context1": case.sample_context1,
                "sample_context2": case.sample_context2,
                **result
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response for {case.lemma}: {e}")
            return {
                "lemma": case.lemma,
                "pos1": case.pos1,
                "pos2": case.pos2,
                "decision": "PARSE_ERROR",
                "confidence": 0.0,
                "correct_pos": None,
                "explanation_ar": "فشل في تحليل الرد",
                "explanation_en": f"Failed to parse response: {response[:200]}",
                "raw_response": response
            }

    def resolve_batch(self, cases: List[ResolutionCase], delay: float = 0.5) -> List[Dict]:
        """Resolve multiple cases with rate limiting"""
        results = []

        for case in tqdm(cases, desc="Resolving with LLM"):
            result = self.resolve_case(case)
            results.append(result)

            # Rate limiting
            time.sleep(delay)

        self.results = results
        return results

    def save_results(self, output_path: str):
        """Save resolution results to CSV and JSON"""
        if not self.results:
            logger.warning("No results to save")
            return

        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        # Save as JSON
        json_file = output_dir / 'llm_resolutions.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved JSON: {json_file}")

        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_file = output_dir / 'llm_resolutions.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Saved CSV: {csv_file}")

        # Generate summary
        self._print_summary(df, output_dir)

    def _print_summary(self, df: pd.DataFrame, output_dir: Path):
        """Print and save summary statistics"""
        print("\n" + "="*80)
        print("LLM RESOLUTION SUMMARY")
        print("="*80)

        # Decision distribution
        decision_counts = df['decision'].value_counts()
        print("\n📊 Decision Distribution:")
        for decision, count in decision_counts.items():
            pct = count / len(df) * 100
            print(f"   {decision:<15} {count:>5} ({pct:>5.1f}%)")

        # Errors found
        errors = df[df['decision'] == 'ERROR']
        if len(errors) > 0:
            print(f"\n🔴 Tagging Errors Found: {len(errors)}")
            print("-"*80)
            for _, row in errors.head(10).iterrows():
                print(f"   {row['lemma']:<15} {row['pos1']} → {row.get('correct_pos', 'N/A')}")
                print(f"      {row.get('explanation_en', '')[:60]}...")

        # Confirmed polysemy
        polysemy = df[df['decision'] == 'POLYSEMY']
        if len(polysemy) > 0:
            print(f"\n🟢 Confirmed Polysemy: {len(polysemy)}")
            print("-"*80)
            for _, row in polysemy.head(10).iterrows():
                print(f"   {row['lemma']:<15} {row['pos1']} / {row['pos2']}")
                print(f"      {row.get('explanation_en', '')[:60]}...")

        # Save summary
        summary = {
            "total_cases": len(df),
            "decisions": decision_counts.to_dict(),
            "errors_found": len(errors),
            "polysemy_confirmed": len(polysemy),
            "uncertain": len(df[df['decision'] == 'UNCERTAIN'])
        }

        summary_file = output_dir / 'llm_resolution_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "="*80)
        print(f"✅ Results saved to: {output_dir}/")
        print("="*80)


def load_validation_results(csv_path: str) -> List[ResolutionCase]:
    """Load validation results and convert to ResolutionCase objects"""
    df = pd.read_csv(csv_path)

    cases = []
    for _, row in df.iterrows():
        case = ResolutionCase(
            lemma=row['lemma'],
            pos1=row['pos1'],
            pos2=row['pos2'],
            sample_words_pos1=str(row.get('sample_words_pos1', '')),
            sample_words_pos2=str(row.get('sample_words_pos2', '')),
            sample_context1=str(row.get('sample_context1', '')),
            sample_context2=str(row.get('sample_context2', '')),
            avg_similarity=float(row.get('avg_similarity', 0)),
            status=str(row.get('status', 'UNKNOWN'))
        )
        cases.append(case)

    return cases


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Resolve lemmatization issues using LLM')
    parser.add_argument('--input', default='validation_reports/validation_results.csv',
                       help='Input CSV with validation results')
    parser.add_argument('--output', default='validation_reports/llm_resolved',
                       help='Output directory for LLM resolutions')
    parser.add_argument('--api-key', default=os.environ.get('OPENROUTER_API_KEY'),
                       help='OpenRouter API key (or set OPENROUTER_API_KEY env var)')
    parser.add_argument('--model', default='deepseek/deepseek-r1',
                       help='Model to use (default: deepseek/deepseek-r1)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of cases to resolve (for testing)')
    parser.add_argument('--filter', choices=['UNCERTAIN', 'LIKELY_ERROR', 'GENUINE_POLYSEMY', 'all'],
                       default='UNCERTAIN', help='Filter cases by status')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API calls (seconds)')

    args = parser.parse_args()

    # Check API key
    if not args.api_key:
        print("❌ Error: OpenRouter API key required")
        print("   Set OPENROUTER_API_KEY environment variable or use --api-key")
        return

    logger.info("="*80)
    logger.info("LLM LEMMATIZATION RESOLVER")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Filter: {args.filter}")

    # Load cases
    cases = load_validation_results(args.input)
    logger.info(f"Loaded {len(cases)} cases from validation results")

    # Filter by status
    if args.filter != 'all':
        cases = [c for c in cases if c.status == args.filter]
        logger.info(f"Filtered to {len(cases)} {args.filter} cases")

    # Limit if specified
    if args.limit:
        cases = cases[:args.limit]
        logger.info(f"Limited to {len(cases)} cases")

    if not cases:
        logger.info("No cases to process")
        return

    # Create resolver and process
    resolver = LemmaResolver(args.api_key, args.model)
    resolver.resolve_batch(cases, delay=args.delay)
    resolver.save_results(args.output)


if __name__ == "__main__":
    main()

