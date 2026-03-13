import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory (project root)
# Assumes this file is at src/lemmatizer/config.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directory
DATA_DIR_ENV = os.getenv("DATA_DIR")
if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).resolve()
else:
    DATA_DIR = BASE_DIR / "data"

# Subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
DB_DIR = DATA_DIR / "db"
LOGS_DIR = BASE_DIR / "logs"

# Specific paths used in the code
DB_TRAINING_DIR = DB_DIR / "db_training"
STOP_WORDS_FILE = DB_TRAINING_DIR / "stop_words/arabic_stop_words.txt"
NON_ARABIC_WORDS_FILE = DB_TRAINING_DIR / "stop_words/non_arabic_words_list.txt"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINTS_DIR, DB_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

