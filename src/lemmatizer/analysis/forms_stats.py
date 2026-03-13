#!/usr/bin/env python3
"""
Comprehensive Lemma Forms Statistics Generator

Analyzes ALL lemmatized documents from camel_lemmatized/ folder and generates
comprehensive statistics about each lemma and its word forms.

Features:
- Lists all forms (surface words) for each lemma
- Calculates statistics: count, mean, median, mode, std dev, quartiles, percentiles
- Generates detailed CSV reports
- Shows distribution analysis (Zipf's law visualization)
- POS distribution per lemma
- Form-level analysis (which forms belong to which lemmas)
- Word diversity metrics
- Advanced frequency analysis
"""

import json
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import statistics
import math
import gc  # For memory management
from lemmatizer.config import PROCESSED_DATA_DIR, DATA_DIR

# Optional imports for advanced stats
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def load_lemmatized_files(lemmatized_dir: Path, limit: int = None) -> tuple:
    """
    Load all lemmatized JSON files and extract word-lemma pairs.

    Returns:
        tuple: (lemma_data, form_data, total_words, total_files)
        - lemma_data: {lemma: {forms: Counter, pos: Counter, contexts: list, files: set}}
        - form_data: {word_form: {lemmas: Counter, pos: Counter}}
    """
    lemma_data = defaultdict(lambda: {
        'forms': Counter(),
        'pos': Counter(),
        'contexts': [],
        'files': set()
    })

    # Also track form -> lemma mapping
    form_data = defaultdict(lambda: {
        'lemmas': Counter(),
        'pos': Counter()
    })

    json_files = list(lemmatized_dir.glob('*_lemmatized.json'))

    # Exclude non-data files like progress.json, summary.json
    json_files = [f for f in json_files if '_lemmatized.json' in f.name]

    if limit:
        json_files = json_files[:limit]

    total_words = 0
    total_files = len(json_files)
    error_files = []
    successful_files = 0

    print(f"Found {total_files} lemmatized files to process...")

    iterator = tqdm(json_files, desc="Loading files", unit="file") if HAS_TQDM else json_files

    for json_file in iterator:
        try:
            # Try reading with utf-8, fallback to utf-8-sig or latin-1
            content = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    with open(json_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                error_files.append((json_file.name, "Encoding error"))
                continue

            # Try to parse JSON, handle truncated files
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                # Try to salvage truncated JSON by finding last complete entry
                if "Extra data" in str(e):
                    # File might have been written twice, take first valid part
                    try:
                        # Find where valid JSON ends
                        pos = e.pos
                        data = json.loads(content[:pos])
                    except:
                        error_files.append((json_file.name, f"JSON error: {e}"))
                        continue
                else:
                    error_files.append((json_file.name, f"JSON error: {e}"))
                    continue

            words_and_lemmas = data.get('words_and_lemmas', [])
            filename = json_file.stem

            for i, item in enumerate(words_and_lemmas):
                word = item.get('word', '')
                lemma = item.get('lemma', '')
                pos = item.get('pos', 'unknown')

                # Skip punctuation and empty entries
                if not lemma or pos == 'punc' or not word.strip():
                    continue

                # Update lemma data
                lemma_data[lemma]['forms'][word] += 1
                lemma_data[lemma]['pos'][pos] += 1
                lemma_data[lemma]['files'].add(filename)

                # Update form data
                form_data[word]['lemmas'][lemma] += 1
                form_data[word]['pos'][pos] += 1

                # Store sample context (first 5 per lemma)
                if len(lemma_data[lemma]['contexts']) < 5:
                    start = max(0, i - 3)
                    end = min(len(words_and_lemmas), i + 4)
                    context_words = [w.get('word', '') for w in words_and_lemmas[start:end]]
                    context = ' '.join(context_words)
                    lemma_data[lemma]['contexts'].append(context)

                total_words += 1

            successful_files += 1

        except Exception as e:
            error_files.append((json_file.name, str(e)))

    print(f"\n✓ Successfully loaded {successful_files:,} files")
    if error_files:
        print(f"⚠ Skipped {len(error_files)} corrupted files")
    print(f"✓ Total words processed: {total_words:,}")
    print(f"✓ Unique lemmas found: {len(lemma_data):,}")
    print(f"✓ Unique word forms found: {len(form_data):,}")

    # Save error log
    if error_files:
        error_log_path = lemmatized_dir.parent / 'lemma_statistics' / 'corrupted_files.txt'
        error_log_path.parent.mkdir(exist_ok=True)
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Corrupted/Skipped Files ({len(error_files)} total)\n")
            f.write("=" * 60 + "\n\n")
            for fname, error in error_files:
                f.write(f"{fname}: {error}\n")
        print(f"✓ Error log saved to: {error_log_path}")

    return dict(lemma_data), dict(form_data), total_words, total_files


def calculate_statistics(lemma_data: dict, form_data: dict, total_words: int, total_files: int) -> tuple:
    """
    Calculate comprehensive statistics for all lemmas and forms.
    """
    stats = {}
    form_stats = {}

    # Calculate per-lemma stats
    lemma_counts = []
    forms_per_lemma = []

    for lemma, data in lemma_data.items():
        total_occurrences = sum(data['forms'].values())
        num_forms = len(data['forms'])
        num_pos = len(data['pos'])

        lemma_counts.append(total_occurrences)
        forms_per_lemma.append(num_forms)

        # Form frequency distribution
        form_freqs = list(data['forms'].values())

        stats[lemma] = {
            'total_occurrences': total_occurrences,
            'unique_forms': num_forms,
            'pos_variants': num_pos,
            'form_list': list(data['forms'].keys()),
            'form_frequencies': dict(data['forms']),
            'pos_distribution': dict(data['pos']),
            'num_files': len(data['files']),
            'sample_contexts': data['contexts'][:3],
            # Form frequency stats
            'form_freq_mean': statistics.mean(form_freqs) if form_freqs else 0,
            'form_freq_median': statistics.median(form_freqs) if form_freqs else 0,
            'form_freq_mode': statistics.mode(form_freqs) if form_freqs else 0,
            'form_freq_std': statistics.stdev(form_freqs) if len(form_freqs) > 1 else 0,
            'form_freq_max': max(form_freqs) if form_freqs else 0,
            'form_freq_min': min(form_freqs) if form_freqs else 0,
        }

    # Calculate per-form stats
    for word_form, data in form_data.items():
        num_lemmas = len(data['lemmas'])
        num_pos = len(data['pos'])
        total_freq = sum(data['lemmas'].values())

        form_stats[word_form] = {
            'total_frequency': total_freq,
            'num_lemmas': num_lemmas,
            'lemmas': dict(data['lemmas']),
            'pos_distribution': dict(data['pos']),
            'num_pos': num_pos,
            'is_ambiguous': num_lemmas > 1
        }

    # Global stats with advanced metrics
    global_stats = {
        'total_files': total_files,
        'total_lemmas': len(lemma_data),
        'total_word_forms': len(form_data),
        'total_occurrences': total_words,
        # Lemma frequency statistics
        'lemma_freq_mean': statistics.mean(lemma_counts) if lemma_counts else 0,
        'lemma_freq_median': statistics.median(lemma_counts) if lemma_counts else 0,
        'lemma_freq_std': statistics.stdev(lemma_counts) if len(lemma_counts) > 1 else 0,
        # Forms per lemma statistics
        'forms_per_lemma_mean': statistics.mean(forms_per_lemma) if forms_per_lemma else 0,
        'forms_per_lemma_median': statistics.median(forms_per_lemma) if forms_per_lemma else 0,
        'forms_per_lemma_max': max(forms_per_lemma) if forms_per_lemma else 0,
        'forms_per_lemma_min': min(forms_per_lemma) if forms_per_lemma else 0,
        # Hapax legomena (lemmas appearing only once)
        'hapax_legomena': sum(1 for c in lemma_counts if c == 1),
        'hapax_ratio': sum(1 for c in lemma_counts if c == 1) / len(lemma_counts) if lemma_counts else 0,
        # Dis legomena (lemmas appearing exactly twice)
        'dis_legomena': sum(1 for c in lemma_counts if c == 2),
        # Distribution analysis
        'top_10_percent_coverage': 0,
        'bottom_50_percent_coverage': 0,
        # Ambiguity stats
        'ambiguous_forms': sum(1 for f in form_stats.values() if f['is_ambiguous']),
        'ambiguity_ratio': sum(1 for f in form_stats.values() if f['is_ambiguous']) / len(form_stats) if form_stats else 0,
    }

    # Calculate coverage (Zipf's law analysis)
    sorted_counts = sorted(lemma_counts, reverse=True)
    total = sum(sorted_counts)
    top_10_idx = max(1, len(sorted_counts) // 10)
    top_20_idx = max(1, len(sorted_counts) // 5)
    bottom_50_idx = len(sorted_counts) // 2

    global_stats['top_10_percent_coverage'] = sum(sorted_counts[:top_10_idx]) / total if total else 0
    global_stats['top_20_percent_coverage'] = sum(sorted_counts[:top_20_idx]) / total if total else 0
    global_stats['bottom_50_percent_coverage'] = sum(sorted_counts[bottom_50_idx:]) / total if total else 0

    # Add quartiles if numpy available
    if HAS_NUMPY:
        global_stats['lemma_freq_q1'] = float(np.percentile(lemma_counts, 25))
        global_stats['lemma_freq_q3'] = float(np.percentile(lemma_counts, 75))
        global_stats['lemma_freq_p90'] = float(np.percentile(lemma_counts, 90))
        global_stats['lemma_freq_p95'] = float(np.percentile(lemma_counts, 95))
        global_stats['lemma_freq_p99'] = float(np.percentile(lemma_counts, 99))

    return stats, form_stats, global_stats


def generate_lemma_forms_table(lemma_data: dict, stats: dict, output_path: Path):
    """
    Generate a detailed CSV table with lemmas and their forms.
    """
    rows = []

    for lemma, data in lemma_data.items():
        lemma_stats = stats[lemma]

        # Sort forms by frequency
        sorted_forms = sorted(data['forms'].items(), key=lambda x: -x[1])

        forms_str = ' | '.join([f"{form}({freq})" for form, freq in sorted_forms[:30]])
        pos_str = ' | '.join([f"{pos}({cnt})" for pos, cnt in sorted(data['pos'].items(), key=lambda x: -x[1])])

        rows.append({
            'lemma': lemma,
            'total_occurrences': lemma_stats['total_occurrences'],
            'unique_forms': lemma_stats['unique_forms'],
            'pos_variants': lemma_stats['pos_variants'],
            'pos_distribution': pos_str,
            'all_forms': forms_str,
            'form_freq_mean': round(lemma_stats['form_freq_mean'], 2),
            'form_freq_median': lemma_stats['form_freq_median'],
            'form_freq_std': round(lemma_stats['form_freq_std'], 2),
            'form_freq_max': lemma_stats['form_freq_max'],
            'form_freq_min': lemma_stats['form_freq_min'],
            'num_files': lemma_stats['num_files'],
            'sample_context': lemma_stats['sample_contexts'][0] if lemma_stats['sample_contexts'] else ''
        })

    # Sort by total occurrences
    rows.sort(key=lambda x: -x['total_occurrences'])

    # Write CSV
    fieldnames = ['lemma', 'total_occurrences', 'unique_forms', 'pos_variants',
                  'pos_distribution', 'all_forms', 'form_freq_mean', 'form_freq_median',
                  'form_freq_std', 'form_freq_max', 'form_freq_min', 'num_files', 'sample_context']

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Lemma forms table saved to: {output_path}")
    return rows


def generate_detailed_forms_csv(lemma_data: dict, output_path: Path):
    """
    Generate a separate CSV with one row per form (word-lemma pair).
    """
    rows = []

    for lemma, data in lemma_data.items():
        for form, freq in data['forms'].items():
            pos_list = list(data['pos'].keys())
            rows.append({
                'word_form': form,
                'lemma': lemma,
                'frequency': freq,
                'pos_options': '|'.join(pos_list),
                'num_pos_variants': len(pos_list)
            })

    # Sort by frequency
    rows.sort(key=lambda x: -x['frequency'])

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word_form', 'lemma', 'frequency', 'pos_options', 'num_pos_variants'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Detailed forms CSV saved to: {output_path}")


def generate_form_ambiguity_report(form_stats: dict, output_path: Path):
    """
    Generate a CSV report of ambiguous forms (forms that map to multiple lemmas).
    """
    rows = []

    for word_form, data in form_stats.items():
        if data['is_ambiguous']:
            lemmas_str = ' | '.join([f"{lemma}({cnt})" for lemma, cnt in
                                      sorted(data['lemmas'].items(), key=lambda x: -x[1])])
            pos_str = ' | '.join([f"{pos}({cnt})" for pos, cnt in
                                   sorted(data['pos_distribution'].items(), key=lambda x: -x[1])])

            rows.append({
                'word_form': word_form,
                'total_frequency': data['total_frequency'],
                'num_lemmas': data['num_lemmas'],
                'lemmas': lemmas_str,
                'num_pos': data['num_pos'],
                'pos_distribution': pos_str
            })

    # Sort by frequency
    rows.sort(key=lambda x: -x['total_frequency'])

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word_form', 'total_frequency', 'num_lemmas',
                                                'lemmas', 'num_pos', 'pos_distribution'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Ambiguous forms report saved to: {output_path}")
    print(f"   Found {len(rows):,} ambiguous forms (mapping to multiple lemmas)")


def generate_frequency_distribution(stats: dict, output_path: Path):
    """
    Generate frequency distribution analysis.
    """
    freq_buckets = Counter()

    for lemma, s in stats.items():
        freq = s['total_occurrences']
        if freq == 1:
            bucket = '1 (hapax)'
        elif freq == 2:
            bucket = '2 (dis)'
        elif freq <= 5:
            bucket = '3-5'
        elif freq <= 10:
            bucket = '6-10'
        elif freq <= 50:
            bucket = '11-50'
        elif freq <= 100:
            bucket = '51-100'
        elif freq <= 500:
            bucket = '101-500'
        elif freq <= 1000:
            bucket = '501-1000'
        else:
            bucket = '1000+'

        freq_buckets[bucket] += 1

    rows = []
    bucket_order = ['1 (hapax)', '2 (dis)', '3-5', '6-10', '11-50', '51-100',
                    '101-500', '501-1000', '1000+']

    total = sum(freq_buckets.values())
    for bucket in bucket_order:
        count = freq_buckets.get(bucket, 0)
        rows.append({
            'frequency_range': bucket,
            'num_lemmas': count,
            'percentage': round(count / total * 100, 2) if total else 0
        })

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frequency_range', 'num_lemmas', 'percentage'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✓ Frequency distribution saved to: {output_path}")


def generate_summary_report(global_stats: dict, stats: dict, output_path: Path):
    """
    Generate a comprehensive summary report.
    """
    # Sort lemmas by occurrence
    sorted_lemmas = sorted(stats.items(), key=lambda x: -x[1]['total_occurrences'])

    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE LEMMA FORMS STATISTICS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    report.append("GLOBAL STATISTICS")
    report.append("-" * 60)
    report.append(f"  Total files analyzed:        {global_stats['total_files']:,}")
    report.append(f"  Total word occurrences:      {global_stats['total_occurrences']:,}")
    report.append(f"  Total unique lemmas:         {global_stats['total_lemmas']:,}")
    report.append(f"  Total unique word forms:     {global_stats['total_word_forms']:,}")
    report.append("")

    report.append("LEMMA FREQUENCY STATISTICS")
    report.append("-" * 60)
    report.append(f"  Mean frequency:              {global_stats['lemma_freq_mean']:.2f}")
    report.append(f"  Median frequency:            {global_stats['lemma_freq_median']:.2f}")
    report.append(f"  Std deviation:               {global_stats['lemma_freq_std']:.2f}")
    if 'lemma_freq_q1' in global_stats:
        report.append(f"  25th percentile (Q1):        {global_stats['lemma_freq_q1']:.2f}")
        report.append(f"  75th percentile (Q3):        {global_stats['lemma_freq_q3']:.2f}")
        report.append(f"  90th percentile:             {global_stats['lemma_freq_p90']:.2f}")
        report.append(f"  95th percentile:             {global_stats['lemma_freq_p95']:.2f}")
        report.append(f"  99th percentile:             {global_stats['lemma_freq_p99']:.2f}")
    report.append("")

    report.append("MORPHOLOGICAL RICHNESS")
    report.append("-" * 60)
    report.append(f"  Forms per lemma (mean):      {global_stats['forms_per_lemma_mean']:.2f}")
    report.append(f"  Forms per lemma (median):    {global_stats['forms_per_lemma_median']:.2f}")
    report.append(f"  Forms per lemma (max):       {global_stats['forms_per_lemma_max']}")
    report.append(f"  Forms per lemma (min):       {global_stats['forms_per_lemma_min']}")
    report.append("")

    report.append("RARE WORDS ANALYSIS")
    report.append("-" * 60)
    report.append(f"  Hapax legomena (freq=1):     {global_stats['hapax_legomena']:,} ({global_stats['hapax_ratio']*100:.1f}%)")
    report.append(f"  Dis legomena (freq=2):       {global_stats['dis_legomena']:,}")
    report.append("")

    report.append("AMBIGUITY ANALYSIS")
    report.append("-" * 60)
    report.append(f"  Ambiguous word forms:        {global_stats['ambiguous_forms']:,}")
    report.append(f"  Ambiguity ratio:             {global_stats['ambiguity_ratio']*100:.1f}%")
    report.append("")

    report.append("ZIPF'S LAW ANALYSIS")
    report.append("-" * 60)
    report.append(f"  Top 10% lemmas coverage:     {global_stats['top_10_percent_coverage']*100:.1f}% of all occurrences")
    if 'top_20_percent_coverage' in global_stats:
        report.append(f"  Top 20% lemmas coverage:     {global_stats['top_20_percent_coverage']*100:.1f}% of all occurrences")
    report.append(f"  Bottom 50% lemmas coverage:  {global_stats['bottom_50_percent_coverage']*100:.1f}% of all occurrences")
    report.append("")

    report.append("TOP 100 MOST FREQUENT LEMMAS")
    report.append("-" * 60)
    report.append(f"{'Rank':<6}{'Lemma':<25}{'Count':>12}{'Forms':>8}{'POS':>6}")
    report.append("-" * 60)

    for i, (lemma, s) in enumerate(sorted_lemmas[:100], 1):
        report.append(f"{i:<6}{lemma:<25}{s['total_occurrences']:>12,}{s['unique_forms']:>8}{s['pos_variants']:>6}")

    report.append("")
    report.append("LEMMAS WITH MOST FORMS (MORPHOLOGICAL RICHNESS)")
    report.append("-" * 80)

    by_forms = sorted(stats.items(), key=lambda x: -x[1]['unique_forms'])[:30]
    report.append(f"{'Lemma':<25}{'Forms':>8}{'Occurrences':>12}{'Top 5 Forms':<50}")
    report.append("-" * 95)

    for lemma, s in by_forms:
        top_forms = list(s['form_frequencies'].keys())[:5]
        report.append(f"{lemma:<25}{s['unique_forms']:>8}{s['total_occurrences']:>12}  {', '.join(top_forms)}")

    report.append("")
    report.append("POS TAG DISTRIBUTION")
    report.append("-" * 60)

    # Aggregate POS across all lemmas
    pos_total = Counter()
    for lemma, data in stats.items():
        for pos, cnt in data['pos_distribution'].items():
            pos_total[pos] += cnt

    total_pos = sum(pos_total.values())
    for pos, cnt in pos_total.most_common():
        pct = cnt / total_pos * 100 if total_pos else 0
        bar = '█' * int(pct / 2)
        report.append(f"  {pos:<20} {cnt:>12,} ({pct:>5.1f}%) {bar}")

    report.append("")
    report.append("LEMMAS WITH MULTIPLE POS (POTENTIAL POLYSEMY)")
    report.append("-" * 60)

    multi_pos = [(l, s) for l, s in stats.items() if s['pos_variants'] > 1]
    multi_pos.sort(key=lambda x: (-x[1]['pos_variants'], -x[1]['total_occurrences']))

    report.append(f"  Total lemmas with 2+ POS tags: {len(multi_pos)}")
    report.append("")

    for lemma, s in multi_pos[:25]:
        pos_str = ', '.join([f"{p}({c})" for p, c in sorted(s['pos_distribution'].items(), key=lambda x: -x[1])])
        report.append(f"  {lemma}: {pos_str}")

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    report_text = '\n'.join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✓ Summary report saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate comprehensive lemma forms statistics')
    parser.add_argument('--input', '-i', default=None,
                        help='Input directory with lemmatized JSON files')
    parser.add_argument('--output', '-o', default=None,
                        help='Output directory for reports')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of files to process (default: all)')

    args = parser.parse_args()

    input_dir = Path(args.input) if args.input else PROCESSED_DATA_DIR / 'camel_lemmatized'
    output_dir = Path(args.output) if args.output else DATA_DIR / 'lemma_statistics'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        sys.exit(1)

    print("=" * 80)
    print("COMPREHENSIVE LEMMA FORMS STATISTICS GENERATOR")
    print("=" * 80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if args.limit:
        print(f"Limit:  {args.limit} files")
    else:
        print(f"Limit:  ALL files (no limit)")
    print()

    # Step 1: Load data
    print("Step 1: Loading all lemmatized files...")
    lemma_data, form_data, total_words, total_files = load_lemmatized_files(input_dir, args.limit)

    if not lemma_data:
        print("No data found!")
        sys.exit(1)

    # Step 2: Calculate statistics
    print("\nStep 2: Calculating comprehensive statistics...")
    stats, form_stats, global_stats = calculate_statistics(lemma_data, form_data, total_words, total_files)

    # Step 3: Generate outputs
    print("\nStep 3: Generating reports...")

    # Main lemma-forms table
    generate_lemma_forms_table(
        lemma_data, stats,
        output_dir / 'lemma_forms_table.csv'
    )

    # Detailed word->lemma mapping
    generate_detailed_forms_csv(
        lemma_data,
        output_dir / 'word_lemma_mapping.csv'
    )

    # Ambiguous forms report
    generate_form_ambiguity_report(
        form_stats,
        output_dir / 'ambiguous_forms.csv'
    )

    # Frequency distribution
    generate_frequency_distribution(
        stats,
        output_dir / 'frequency_distribution.csv'
    )

    # Summary report
    generate_summary_report(
        global_stats, stats,
        output_dir / 'statistics_summary.txt'
    )

    # Save raw stats as JSON for further analysis
    print("\nStep 4: Saving full statistics as JSON...")
    json_stats = {
        'global': global_stats,
        'per_lemma': {k: {
            'total_occurrences': v['total_occurrences'],
            'unique_forms': v['unique_forms'],
            'pos_variants': v['pos_variants'],
            'form_frequencies': v['form_frequencies'],
            'pos_distribution': v['pos_distribution']
        } for k, v in stats.items()}
    }

    with open(output_dir / 'statistics_full.json', 'w', encoding='utf-8') as f:
        json.dump(json_stats, f, ensure_ascii=False, indent=2)

    print(f"✓ Full statistics JSON saved to: {output_dir / 'statistics_full.json'}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files generated in: {output_dir}/")
    print("  - lemma_forms_table.csv       (All lemmas with their forms)")
    print("  - word_lemma_mapping.csv      (All word->lemma pairs)")
    print("  - ambiguous_forms.csv         (Forms mapping to multiple lemmas)")
    print("  - frequency_distribution.csv  (Frequency bucket analysis)")
    print("  - statistics_summary.txt      (Human-readable summary)")
    print("  - statistics_full.json        (Complete JSON data)")
    print()


if __name__ == '__main__':
    main()

