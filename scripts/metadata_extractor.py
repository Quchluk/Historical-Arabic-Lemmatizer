#!/usr/bin/env python3
"""
OpenITI Metadata Extractor
Extracts metadata from OpenITI YAML files and creates a comprehensive Parquet file.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import pandas as pd
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AuthorMetadata:
    """Author-level metadata"""
    author_uri: str = ""
    time_period: str = ""
    ism_ar: str = ""
    kunya_ar: str = ""
    laqab_ar: str = ""
    nasab_ar: str = ""
    nisba_ar: str = ""
    shuhra_ar: str = ""
    birth_place: str = ""
    death_place: str = ""
    resided_places: str = ""
    visited_places: str = ""
    birth_date_ah: str = ""
    death_date_ah: str = ""
    students: str = ""
    teachers: str = ""
    external_ids: str = ""
    bibliography: str = ""
    comments: str = ""
    file_path: str = ""


@dataclass
class BookMetadata:
    """Book-level metadata"""
    book_uri: str = ""
    author_uri: str = ""
    time_period: str = ""
    genres: str = ""
    title_a_ar: str = ""
    title_b_ar: str = ""
    writing_place: str = ""
    writing_date_ah: str = ""
    related_books: str = ""
    external_ids: str = ""
    editions: str = ""
    links: str = ""
    manuscripts: str = ""
    studies: str = ""
    translations: str = ""
    comments: str = ""
    file_path: str = ""


@dataclass
class VersionMetadata:
    """Version-level metadata"""
    version_uri: str = ""
    book_uri: str = ""
    author_uri: str = ""
    time_period: str = ""
    length_words: int = 0
    length_chars: int = 0
    based_on: str = ""
    collated_with: str = ""
    links: str = ""
    annotator: str = ""
    annotation_date: str = ""
    issues: str = ""
    comments: str = ""
    text_file_path: str = ""
    metadata_file_path: str = ""
    # Embedded text file metadata
    embedded_author: str = ""
    embedded_book_title: str = ""
    embedded_bkid: str = ""
    embedded_shamela_record: str = ""
    embedded_auth: str = ""
    embedded_lng: str = ""
    embedded_max: str = ""
    embedded_authinf: str = ""
    embedded_ndata: str = ""
    embedded_bkord: str = ""
    embedded_archive: str = ""
    embedded_authno: str = ""
    embedded_ad: str = ""
    embedded_idx: str = ""
    embedded_comp: str = ""
    embedded_higrid: str = ""
    embedded_cat: str = ""
    embedded_iso: str = ""
    embedded_download_source: str = ""
    embedded_download_date: str = ""
    embedded_conversion_date: str = ""
    embedded_volumes: str = ""
    embedded_publisher: str = ""
    embedded_edition_year: str = ""
    embedded_links: str = ""
    embedded_other: str = ""


class YAMLParser:
    """Parse OpenITI YAML metadata files"""

    @staticmethod
    def parse_yaml_file(file_path: str) -> Dict[str, str]:
        """Parse a YAML file and extract metadata fields"""
        metadata = {}
        raw_content = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                raw_content = content  # Keep for debugging

            # Parse line by line to handle multi-line values
            current_key = None
            current_value = []

            for line in content.split('\n'):
                # Check if line starts with a field code (more flexible pattern)
                match = re.match(r'^(\d+#[A-Z]+#[A-Z#]+):\s*(.*)', line)
                if match:
                    # Save previous field
                    if current_key:
                        metadata[current_key] = ' '.join(current_value).strip()

                    current_key = match.group(1)
                    current_value = [match.group(2)]
                elif current_key and line.strip() and not line.startswith('#'):
                    # Continuation of previous field (skip comment-only lines)
                    current_value.append(line.strip())

            # Save last field
            if current_key:
                metadata[current_key] = ' '.join(current_value).strip()

            # Store raw content for any field we might have missed
            metadata['__RAW_CONTENT__'] = raw_content

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return metadata

    @staticmethod
    def parse_embedded_text_metadata(file_path: str) -> Dict[str, str]:
        """Parse embedded metadata from OpenITI text files"""
        metadata = {}
        other_metadata = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                in_header = False

                for line in f:
                    line = line.strip()

                    # Check for header start
                    if line.startswith('######OpenITI#'):
                        in_header = True
                        continue

                    # Check for header end
                    if line.startswith('#META#Header#End#'):
                        break

                    # Parse metadata lines
                    if in_header and line.startswith('#META#'):
                        # Remove #META# prefix
                        content = line[6:].strip()

                        # Try to parse key :: value or key: value format
                        separator = ' :: ' if ' :: ' in content else ':'
                        if separator in content:
                            parts = content.split(separator, 1)
                            key = parts[0].strip()
                            value = parts[1].strip() if len(parts) > 1 else ''

                            # Skip NODATA, NOTGIVEN, NOCODE values
                            if value in ['NODATA', 'NOTGIVEN', 'NOCODE', '']:
                                continue

                            # Map known keys to standardized names
                            # Old format (simple keys)
                            if key == 'المؤلف' or key == '010.AuthorNAME':
                                metadata['author'] = value
                            elif key == 'bkid' or key == '000.SortField':
                                # Extract book ID from Shamela_XXXXXXX format
                                if value.startswith('Shamela_'):
                                    metadata['bkid'] = value.replace('Shamela_', '')
                                else:
                                    metadata['bkid'] = value
                            elif key == 'Shamela_short_metadata_record':
                                metadata['shamela_record'] = value
                            elif key == 'ملاحظة':
                                if 'shamela_record' in metadata:
                                    metadata['shamela_record'] += ' | ' + value
                                else:
                                    metadata['shamela_record'] = value
                            elif key == 'auth':
                                metadata['auth'] = value
                            elif key == 'Lng':
                                metadata['lng'] = value
                            elif key == 'max':
                                metadata['max'] = value
                            elif key == 'bk' or key == '020.BookTITLE':
                                metadata['book_title'] = value
                            elif key == '021.BookSUBJ':
                                metadata['cat'] = value
                            elif key == 'authinf' or key == '010.AuthorAKA':
                                if value and 'authinf' not in metadata:
                                    metadata['authinf'] = value
                            elif key == 'ndata':
                                metadata['ndata'] = value
                            elif key == 'bkord':
                                metadata['bkord'] = value
                            elif key == 'archive':
                                metadata['archive'] = value
                            elif key == 'الكتاب':
                                if 'book_title' not in metadata:
                                    metadata['book_title'] = value
                            elif key == 'authno':
                                metadata['authno'] = value
                            elif key == 'ad':
                                metadata['ad'] = value
                            elif key == 'idx':
                                metadata['idx'] = value
                            elif key == 'comp':
                                metadata['comp'] = value
                            elif key == 'higrid' or key == '011.AuthorDIED':
                                metadata['higrid'] = value
                            elif key == 'cat':
                                metadata['cat'] = value
                            elif key == 'iso':
                                metadata['iso'] = value
                            elif key == 'DownloadSource' or key == '030.LibURI':
                                metadata['download_source'] = value
                            elif key == 'DownloadDate':
                                metadata['download_date'] = value
                            elif key == 'ConversionDate':
                                metadata['conversion_date'] = value
                            elif key == '031.LibURL' or key == '031.LibREADONLINE':
                                if 'links' not in metadata:
                                    metadata['links'] = value
                                else:
                                    metadata['links'] += ' | ' + value
                            elif key == '022.BookVOLS':
                                metadata['volumes'] = value
                            elif key == '025.BookLANG':
                                metadata['lng'] = value
                            elif key == '043.EdPUBLISHER':
                                metadata['publisher'] = value
                            elif key == '045.EdYEAR':
                                metadata['edition_year'] = value
                            else:
                                other_metadata.append(f"{key}: {value}")
                        else:
                            # Non-key:value metadata
                            if content:
                                other_metadata.append(content)

        except Exception as e:
            logger.error(f"Error parsing embedded metadata from {file_path}: {e}")

        # Combine other metadata
        if other_metadata:
            metadata['other'] = ' | '.join(other_metadata)

        return metadata

    @staticmethod
    def extract_author_metadata(yaml_data: Dict[str, str], file_path: str, time_period: str) -> AuthorMetadata:
        """Extract author metadata from parsed YAML"""
        return AuthorMetadata(
            author_uri=yaml_data.get('00#AUTH#URI######', ''),
            time_period=time_period,
            ism_ar=yaml_data.get('10#AUTH#ISM####AR', ''),
            kunya_ar=yaml_data.get('10#AUTH#KUNYA##AR', ''),
            laqab_ar=yaml_data.get('10#AUTH#LAQAB##AR', ''),
            nasab_ar=yaml_data.get('10#AUTH#NASAB##AR', ''),
            nisba_ar=yaml_data.get('10#AUTH#NISBA##AR', ''),
            shuhra_ar=yaml_data.get('10#AUTH#SHUHRA#AR', ''),
            birth_place=yaml_data.get('20#AUTH#BORN#####', ''),
            death_place=yaml_data.get('20#AUTH#DIED#####', ''),
            resided_places=yaml_data.get('20#AUTH#RESIDED##', ''),
            visited_places=yaml_data.get('20#AUTH#VISITED##', ''),
            birth_date_ah=yaml_data.get('30#AUTH#BORN###AH', ''),
            death_date_ah=yaml_data.get('30#AUTH#DIED###AH', ''),
            students=yaml_data.get('40#AUTH#STUDENTS#', ''),
            teachers=yaml_data.get('40#AUTH#TEACHERS#', ''),
            external_ids=yaml_data.get('70#AUTH#EXTID####', ''),
            bibliography=yaml_data.get('80#AUTH#BIBLIO###', ''),
            comments=yaml_data.get('90#AUTH#COMMENT##', ''),
            file_path=file_path
        )

    @staticmethod
    def extract_book_metadata(yaml_data: Dict[str, str], file_path: str, author_uri: str, time_period: str) -> BookMetadata:
        """Extract book metadata from parsed YAML"""
        return BookMetadata(
            book_uri=yaml_data.get('00#BOOK#URI######', ''),
            author_uri=author_uri,
            time_period=time_period,
            genres=yaml_data.get('10#BOOK#GENRES###', ''),
            title_a_ar=yaml_data.get('10#BOOK#TITLEA#AR', ''),
            title_b_ar=yaml_data.get('10#BOOK#TITLEB#AR', ''),
            writing_place=yaml_data.get('20#BOOK#WROTE####', ''),
            writing_date_ah=yaml_data.get('30#BOOK#WROTE##AH', ''),
            related_books=yaml_data.get('40#BOOK#RELATED##', ''),
            external_ids=yaml_data.get('70#BOOK#EXTID####', ''),
            editions=yaml_data.get('80#BOOK#EDITIONS#', ''),
            links=yaml_data.get('80#BOOK#LINKS####', ''),
            manuscripts=yaml_data.get('80#BOOK#MSS######', ''),
            studies=yaml_data.get('80#BOOK#STUDIES##', ''),
            translations=yaml_data.get('80#BOOK#TRANSLAT#', ''),
            comments=yaml_data.get('90#BOOK#COMMENT##', ''),
            file_path=file_path
        )

    @staticmethod
    def extract_version_metadata(yaml_data: Dict[str, str], file_path: str, book_uri: str, author_uri: str, time_period: str) -> VersionMetadata:
        """Extract version metadata from parsed YAML"""
        # Parse integer fields safely
        length_words = 0
        length_chars = 0
        try:
            length_words = int(yaml_data.get('00#VERS#LENGTH###', '0'))
        except (ValueError, TypeError):
            pass
        try:
            length_chars = int(yaml_data.get('00#VERS#CLENGTH##', '0'))
        except (ValueError, TypeError):
            pass

        # Find corresponding text file
        text_file_path = ""
        version_uri = yaml_data.get('00#VERS#URI######', '')
        if version_uri:
            # Text file has same name without .yml extension
            potential_text_file = file_path.replace('.yml', '')
            if os.path.exists(potential_text_file):
                text_file_path = potential_text_file

        # Extract embedded metadata from text file if it exists
        embedded_meta = {}
        if text_file_path and os.path.exists(text_file_path):
            embedded_meta = YAMLParser.parse_embedded_text_metadata(text_file_path)

        return VersionMetadata(
            version_uri=version_uri,
            book_uri=book_uri,
            author_uri=author_uri,
            time_period=time_period,
            length_words=length_words,
            length_chars=length_chars,
            based_on=yaml_data.get('80#VERS#BASED####', ''),
            collated_with=yaml_data.get('80#VERS#COLLATED#', ''),
            links=yaml_data.get('80#VERS#LINKS####', ''),
            annotator=yaml_data.get('90#VERS#ANNOTATOR', ''),
            annotation_date=yaml_data.get('90#VERS#DATE#####', ''),
            issues=yaml_data.get('90#VERS#ISSUES###', ''),
            comments=yaml_data.get('90#VERS#COMMENT##', ''),
            text_file_path=text_file_path,
            metadata_file_path=file_path,
            # Embedded metadata from text file
            embedded_author=embedded_meta.get('author', ''),
            embedded_book_title=embedded_meta.get('book_title', ''),
            embedded_bkid=embedded_meta.get('bkid', ''),
            embedded_shamela_record=embedded_meta.get('shamela_record', ''),
            embedded_auth=embedded_meta.get('auth', ''),
            embedded_lng=embedded_meta.get('lng', ''),
            embedded_max=embedded_meta.get('max', ''),
            embedded_authinf=embedded_meta.get('authinf', ''),
            embedded_ndata=embedded_meta.get('ndata', ''),
            embedded_bkord=embedded_meta.get('bkord', ''),
            embedded_archive=embedded_meta.get('archive', ''),
            embedded_authno=embedded_meta.get('authno', ''),
            embedded_ad=embedded_meta.get('ad', ''),
            embedded_idx=embedded_meta.get('idx', ''),
            embedded_comp=embedded_meta.get('comp', ''),
            embedded_higrid=embedded_meta.get('higrid', ''),
            embedded_cat=embedded_meta.get('cat', ''),
            embedded_iso=embedded_meta.get('iso', ''),
            embedded_download_source=embedded_meta.get('download_source', ''),
            embedded_download_date=embedded_meta.get('download_date', ''),
            embedded_conversion_date=embedded_meta.get('conversion_date', ''),
            embedded_volumes=embedded_meta.get('volumes', ''),
            embedded_publisher=embedded_meta.get('publisher', ''),
            embedded_edition_year=embedded_meta.get('edition_year', ''),
            embedded_links=embedded_meta.get('links', ''),
            embedded_other=embedded_meta.get('other', '')
        )


class MetadataExtractor:
    """Main metadata extraction orchestrator"""

    def __init__(self, db_root: str):
        self.db_root = Path(db_root)
        self.parser = YAMLParser()
        self.authors: List[AuthorMetadata] = []
        self.books: List[BookMetadata] = []
        self.versions: List[VersionMetadata] = []
        self.stats = defaultdict(int)

    def scan_directory(self) -> None:
        """Recursively scan the database directory and extract metadata"""
        logger.info(f"Starting scan of {self.db_root}")

        # Find all time period folders
        time_period_folders = sorted([d for d in self.db_root.iterdir()
                                     if d.is_dir() and re.match(r'^\d{4}AH$', d.name)])

        logger.info(f"Found {len(time_period_folders)} time period folders")

        for period_folder in tqdm(time_period_folders, desc="Processing time periods"):
            time_period = period_folder.name
            data_folder = period_folder / 'data'

            if not data_folder.exists():
                continue

            # Find all author folders
            author_folders = sorted([d for d in data_folder.iterdir() if d.is_dir()])

            for author_folder in tqdm(author_folders, desc=f"  {time_period}", leave=False):
                self.process_author_folder(author_folder, time_period)

        logger.info("Scan complete!")
        self.log_statistics()

    def process_author_folder(self, author_folder: Path, time_period: str) -> None:
        """Process a single author folder"""
        author_id = author_folder.name
        author_yml = author_folder / f"{author_id}.yml"

        # Extract author metadata
        if author_yml.exists():
            yaml_data = self.parser.parse_yaml_file(str(author_yml))
            author_metadata = self.parser.extract_author_metadata(
                yaml_data, str(author_yml), time_period
            )
            self.authors.append(author_metadata)
            self.stats['authors'] += 1
        else:
            logger.warning(f"Missing author YML: {author_yml}")
            return

        # Find all book folders
        book_folders = [d for d in author_folder.iterdir()
                       if d.is_dir() and d.name.startswith(author_id)]

        for book_folder in book_folders:
            self.process_book_folder(book_folder, author_id, time_period)

    def process_book_folder(self, book_folder: Path, author_uri: str, time_period: str) -> None:
        """Process a single book folder"""
        book_id = book_folder.name
        book_yml = book_folder / f"{book_id}.yml"

        # Extract book metadata
        if book_yml.exists():
            yaml_data = self.parser.parse_yaml_file(str(book_yml))
            book_metadata = self.parser.extract_book_metadata(
                yaml_data, str(book_yml), author_uri, time_period
            )
            self.books.append(book_metadata)
            self.stats['books'] += 1
        else:
            logger.warning(f"Missing book YML: {book_yml}")
            # Don't return - still process text files
            book_id = book_folder.name  # Use folder name as book URI

        # Find all version YAML files (exclude the book-level YML and non-YML files)
        version_ymls = [f for f in book_folder.iterdir()
                       if f.is_file() and f.suffix == '.yml' and f.name != f"{book_id}.yml"]

        for version_yml in version_ymls:
            self.process_version_file(version_yml, book_id, author_uri, time_period)

        # Also process text files without corresponding YML files
        # Get all text files (files without .yml extension)
        all_files = [f for f in book_folder.iterdir() if f.is_file() and f.suffix != '.yml']

        for text_file in all_files:
            # Check if there's a corresponding .yml file
            yml_file = book_folder / f"{text_file.name}.yml"
            if not yml_file.exists():
                # Process standalone text file
                self.process_standalone_text_file(text_file, book_id, author_uri, time_period)

    def process_version_file(self, version_yml: Path, book_uri: str, author_uri: str, time_period: str) -> None:
        """Process a single version YAML file"""
        yaml_data = self.parser.parse_yaml_file(str(version_yml))
        version_metadata = self.parser.extract_version_metadata(
            yaml_data, str(version_yml), book_uri, author_uri, time_period
        )
        self.versions.append(version_metadata)
        self.stats['versions'] += 1

    def process_standalone_text_file(self, text_file: Path, book_uri: str, author_uri: str, time_period: str) -> None:
        """Process a text file that doesn't have a corresponding YML file"""
        # Extract embedded metadata from the text file
        embedded_meta = self.parser.parse_embedded_text_metadata(str(text_file))

        # Create version metadata from embedded data
        version_metadata = VersionMetadata(
            version_uri=text_file.name,  # Use filename as URI
            book_uri=book_uri,
            author_uri=author_uri,
            time_period=time_period,
            length_words=0,  # Would need to count if needed
            length_chars=0,  # Would need to count if needed
            based_on='',
            collated_with='',
            links='',
            annotator='',
            annotation_date='',
            issues='',
            comments='',
            text_file_path=str(text_file),
            metadata_file_path='',  # No YML file
            # Embedded metadata from text file
            embedded_author=embedded_meta.get('author', ''),
            embedded_book_title=embedded_meta.get('book_title', ''),
            embedded_bkid=embedded_meta.get('bkid', ''),
            embedded_shamela_record=embedded_meta.get('shamela_record', ''),
            embedded_auth=embedded_meta.get('auth', ''),
            embedded_lng=embedded_meta.get('lng', ''),
            embedded_max=embedded_meta.get('max', ''),
            embedded_authinf=embedded_meta.get('authinf', ''),
            embedded_ndata=embedded_meta.get('ndata', ''),
            embedded_bkord=embedded_meta.get('bkord', ''),
            embedded_archive=embedded_meta.get('archive', ''),
            embedded_authno=embedded_meta.get('authno', ''),
            embedded_ad=embedded_meta.get('ad', ''),
            embedded_idx=embedded_meta.get('idx', ''),
            embedded_comp=embedded_meta.get('comp', ''),
            embedded_higrid=embedded_meta.get('higrid', ''),
            embedded_cat=embedded_meta.get('cat', ''),
            embedded_iso=embedded_meta.get('iso', ''),
            embedded_download_source=embedded_meta.get('download_source', ''),
            embedded_download_date=embedded_meta.get('download_date', ''),
            embedded_conversion_date=embedded_meta.get('conversion_date', ''),
            embedded_volumes=embedded_meta.get('volumes', ''),
            embedded_publisher=embedded_meta.get('publisher', ''),
            embedded_edition_year=embedded_meta.get('edition_year', ''),
            embedded_links=embedded_meta.get('links', ''),
            embedded_other=embedded_meta.get('other', '')
        )

        self.versions.append(version_metadata)
        self.stats['versions'] += 1
        self.stats['standalone_text_files'] += 1

    def log_statistics(self) -> None:
        """Log extraction statistics"""
        logger.info("=" * 60)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Authors extracted:  {self.stats['authors']:,}")
        logger.info(f"Books extracted:    {self.stats['books']:,}")
        logger.info(f"Versions extracted: {self.stats['versions']:,}")
        if self.stats.get('standalone_text_files', 0) > 0:
            logger.info(f"  - From standalone text files: {self.stats['standalone_text_files']:,}")
        logger.info("=" * 60)

    def create_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert extracted metadata to pandas DataFrames"""
        logger.info("Creating DataFrames...")

        authors_df = pd.DataFrame([asdict(a) for a in self.authors])
        books_df = pd.DataFrame([asdict(b) for b in self.books])
        versions_df = pd.DataFrame([asdict(v) for v in self.versions])

        return authors_df, books_df, versions_df

    def create_combined_dataframe(self) -> pd.DataFrame:
        """Create a single combined DataFrame with all metadata"""
        logger.info("Creating combined DataFrame...")

        authors_df, books_df, versions_df = self.create_dataframes()

        # Merge all three levels
        # First merge books with authors
        combined_df = versions_df.merge(
            books_df,
            on=['book_uri', 'author_uri', 'time_period'],
            how='left',
            suffixes=('_version', '_book')
        )

        # Then merge with authors
        combined_df = combined_df.merge(
            authors_df,
            on=['author_uri', 'time_period'],
            how='left',
            suffixes=('', '_author')
        )

        # Reorder columns for better readability
        priority_cols = [
            'time_period',
            'author_uri',
            'book_uri',
            'version_uri',
            'ism_ar',
            'shuhra_ar',
            'title_a_ar',
            'birth_date_ah',
            'death_date_ah',
            'length_words',
            'length_chars'
        ]

        other_cols = [col for col in combined_df.columns if col not in priority_cols]
        combined_df = combined_df[priority_cols + other_cols]

        logger.info(f"Combined DataFrame shape: {combined_df.shape}")

        return combined_df

    def save_to_parquet(self, output_file: str, combined: bool = True) -> None:
        """Save metadata to Parquet file(s)"""
        if combined:
            # Save single combined file
            logger.info(f"Saving combined metadata to {output_file}")
            df = self.create_combined_dataframe()
            df.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
            logger.info(f"✓ Saved {len(df):,} records to {output_file}")

        else:
            # Save separate files for each level
            authors_df, books_df, versions_df = self.create_dataframes()

            base_name = output_file.rsplit('.', 1)[0]

            authors_file = f"{base_name}_authors.parquet"
            books_file = f"{base_name}_books.parquet"
            versions_file = f"{base_name}_versions.parquet"

            logger.info(f"Saving authors to {authors_file}")
            authors_df.to_parquet(authors_file, engine='pyarrow', compression='snappy', index=False)

            logger.info(f"Saving books to {books_file}")
            books_df.to_parquet(books_file, engine='pyarrow', compression='snappy', index=False)

            logger.info(f"Saving versions to {versions_file}")
            versions_df.to_parquet(versions_file, engine='pyarrow', compression='snappy', index=False)

            logger.info(f"✓ Saved authors: {len(authors_df):,}, books: {len(books_df):,}, versions: {len(versions_df):,}")

    def save_to_csv(self, output_file: str, combined: bool = True) -> None:
        """Save metadata to CSV file(s) for easy inspection"""
        if combined:
            logger.info(f"Saving combined metadata to CSV: {output_file}")
            df = self.create_combined_dataframe()
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"✓ Saved {len(df):,} records to {output_file}")
        else:
            authors_df, books_df, versions_df = self.create_dataframes()

            base_name = output_file.rsplit('.', 1)[0]

            authors_file = f"{base_name}_authors.csv"
            books_file = f"{base_name}_books.csv"
            versions_file = f"{base_name}_versions.csv"

            authors_df.to_csv(authors_file, index=False, encoding='utf-8')
            books_df.to_csv(books_file, index=False, encoding='utf-8')
            versions_df.to_csv(versions_file, index=False, encoding='utf-8')

            logger.info(f"✓ Saved CSV files: authors, books, versions")

    def generate_summary_report(self, output_file: str = 'metadata_summary.txt') -> None:
        """Generate a summary report of the extracted metadata"""
        logger.info(f"Generating summary report: {output_file}")

        authors_df, books_df, versions_df = self.create_dataframes()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("OpenITI Metadata Extraction Summary Report\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Authors:  {len(authors_df):,}\n")
            f.write(f"Total Books:    {len(books_df):,}\n")
            f.write(f"Total Versions: {len(versions_df):,}\n\n")

            # By time period
            f.write("DISTRIBUTION BY TIME PERIOD\n")
            f.write("-" * 80 + "\n")
            period_stats = authors_df['time_period'].value_counts().sort_index()
            for period, count in period_stats.items():
                f.write(f"{period}: {count:,} authors\n")
            f.write("\n")

            # Field completion rates
            f.write("AUTHOR FIELD COMPLETION RATES\n")
            f.write("-" * 80 + "\n")
            for col in authors_df.columns:
                if col not in ['file_path', 'author_uri', 'time_period']:
                    non_empty = (authors_df[col].astype(str).str.strip() != '').sum()
                    pct = (non_empty / len(authors_df) * 100) if len(authors_df) > 0 else 0
                    f.write(f"{col:25s}: {pct:5.1f}% ({non_empty:,}/{len(authors_df):,})\n")
            f.write("\n")

            # External IDs
            f.write("EXTERNAL IDs COVERAGE\n")
            f.write("-" * 80 + "\n")
            authors_with_extid = (authors_df['external_ids'].astype(str).str.strip() != '').sum()
            books_with_extid = (books_df['external_ids'].astype(str).str.strip() != '').sum()
            f.write(f"Authors with external IDs: {authors_with_extid:,} ({authors_with_extid/len(authors_df)*100:.1f}%)\n")
            f.write(f"Books with external IDs:   {books_with_extid:,} ({books_with_extid/len(books_df)*100:.1f}%)\n\n")

            # Text statistics
            if len(versions_df) > 0:
                f.write("TEXT STATISTICS\n")
                f.write("-" * 80 + "\n")
                total_words = versions_df['length_words'].sum()
                total_chars = versions_df['length_chars'].sum()
                f.write(f"Total words across all versions:      {total_words:,}\n")
                f.write(f"Total characters across all versions: {total_chars:,}\n")
                f.write(f"Average words per version:            {total_words/len(versions_df):,.0f}\n")
                f.write(f"Average characters per version:       {total_chars/len(versions_df):,.0f}\n")

        logger.info(f"✓ Summary report saved to {output_file}")


def main():
    """Main entry point"""
    import argparse
    from lemmatizer.config import RAW_DATA_DIR, DB_DIR

    parser = argparse.ArgumentParser(description='Extract OpenITI metadata to Parquet format')
    parser.add_argument('--db-root', default=str(DB_DIR / 'db'), help='Root directory of OpenITI database')
    parser.add_argument('--output', default=str(RAW_DATA_DIR / 'openiti_metadata.parquet'), help='Output Parquet file path')
    parser.add_argument('--csv', action='store_true', help='Also save as CSV')
    parser.add_argument('--separate', action='store_true', help='Save separate files for authors, books, and versions')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')

    args = parser.parse_args()

    # Initialize extractor
    extractor = MetadataExtractor(args.db_root)

    # Scan and extract
    extractor.scan_directory()

    # Save to Parquet
    extractor.save_to_parquet(args.output, combined=not args.separate)

    # Optionally save to CSV
    if args.csv:
        csv_output = args.output.replace('.parquet', '.csv')
        extractor.save_to_csv(csv_output, combined=not args.separate)

    # Optionally generate summary
    if args.summary:
        extractor.generate_summary_report()

    logger.info("✓ All done!")


if __name__ == '__main__':
    main()

