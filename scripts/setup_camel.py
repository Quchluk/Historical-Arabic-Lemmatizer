#!/usr/bin/env python3
"""
Download CAMeL Tools data and setup
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("CAMeL TOOLS DATA SETUP")
print("="*80)

# Method 1: Try using camel_data command
print("\nMethod 1: Trying camel_data command...")
try:
    result = subprocess.run(['camel_data', '--help'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ camel_data command found")

        print("\nDownloading light morphology data...")
        result = subprocess.run(
            ['camel_data', '-i', 'light'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ Light data downloaded")
            print(result.stdout)
        else:
            print("✗ Download failed:", result.stderr)
    else:
        print("✗ camel_data command not working")
except FileNotFoundError:
    print("✗ camel_data command not found in PATH")

# Method 2: Try Python API
print("\nMethod 2: Trying Python API...")
try:
    from camel_tools.data import DataCatalogue

    catalogue = DataCatalogue.builtin_catalogue()
    print(f"✓ Data catalogue loaded")

    # Try to install morphology database
    print("\nAttempting to download morphology database...")
    try:
        from camel_tools.cli.camel_data import main as camel_data_main

        # Simulate command line args
        sys.argv = ['camel_data', '-i', 'light']
        camel_data_main()
        print("✓ Data download initiated")
    except Exception as e:
        print(f"Note: {e}")

except ImportError as e:
    print(f"✗ Import failed: {e}")

# Method 3: Manual installation instructions
print("\n" + "="*80)
print("MANUAL INSTALLATION METHOD")
print("="*80)
print("""
If automatic download fails, install manually:

1. Run this command in terminal:
   pip install camel-tools[all]

2. Then download data:
   camel_data -i light
   
   OR for full data:
   camel_data -i all

3. If camel_data command not found, try:
   python -m camel_tools.cli.camel_data -i light

4. Verify installation:
   python -c "from camel_tools.morphology.database import MorphologyDB; db = MorphologyDB.builtin_db('calima-msa-s31')"
""")

# Test if data already exists
print("\n" + "="*80)
print("TESTING CURRENT INSTALLATION")
print("="*80)

try:
    print("\nTrying to load MSA database...")
    from camel_tools.morphology.database import MorphologyDB

    # Try different database names
    db_names = ['calima-msa-s31', 'msa', 'MSA']

    for db_name in db_names:
        try:
            print(f"  Trying '{db_name}'...")
            db = MorphologyDB.builtin_db(db_name)
            print(f"  ✓ SUCCESS! Database '{db_name}' loaded!")

            # Test analyzer
            from camel_tools.morphology.analyzer import Analyzer
            analyzer = Analyzer(db)
            print(f"  ✓ Analyzer initialized")

            # Quick test
            word = 'كتاب'
            analyses = analyzer.analyze(word)
            print(f"  ✓ Test word '{word}' analyzed: {len(analyses)} analyses found")

            print(f"\n✅ CAMeL Tools is READY with database: {db_name}")
            print(f"You can now run: python camel_lemmatize_db.py --limit 1")
            sys.exit(0)

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n❌ No working database found. Need to download data.")
    print("Run: python -m camel_tools.cli.camel_data -i light")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nCAMeL Tools not properly installed.")
    print("Try: pip install camel-tools[all]")

