"""
Enhanced RAG Integration with Web Search Fallback
Provides Wikipedia-style citations with automatic web search when RAG returns no results
Supports full PDF access via S3 storage
"""

from typing import List, Dict, Any, Optional, Tuple
from app.services.research import search_research_chunks_from_text
from app.services.comparative_analysis import safe_generate
import re
import os
import io
import logging
import requests
from dotenv import load_dotenv
from tavily import TavilyClient
from bs4 import BeautifulSoup
from datetime import datetime
import pdfplumber
from PIL import Image
import pytesseract

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available - OCR disabled")

logger = logging.getLogger(__name__)

load_dotenv()

logger.setLevel(logging.DEBUG)

tavily_api_key = os.getenv('TAVILY_API_KEY')
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


def extract_author_from_webpage(url: str, title: str = "") -> Dict[str, Optional[str]]:
    """
    Extract author name and publication year from webpage HTML
    Returns author in "FirstName L." format (e.g., "Erin N.")
    
    Args:
        url: Webpage URL to fetch and parse
        title: Page title from search results (fallback)
    
    Returns:
        {
            'author': 'FirstName L.' or None,
            'year': 'YYYY' or 'n.d.',
            'organization': 'Domain-based org name'
        }
    """
    try:        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        author_name = None
        pub_year = None
        
        meta_author = soup.find('meta', attrs={'name': 'author'})
        if meta_author and meta_author.get('content'):
            author_name = meta_author.get('content').strip()
        
        if not author_name:
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    import json
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        author_data = data.get('author', {})
                        if isinstance(author_data, dict):
                            author_name = author_data.get('name')
                        elif isinstance(author_data, str):
                            author_name = author_data
                        if author_name:
                            break
                except:
                    continue
        
        if not author_name:
            author_patterns = [
                ('class', 'author'),
                ('class', 'writer'),
                ('class', 'byline'),
                ('rel', 'author'),
                ('itemprop', 'author'),
                ('itemprop', 'name')
            ]
            
            for attr_name, attr_value in author_patterns:
                author_elem = soup.find(attrs={attr_name: lambda x: x and attr_value in str(x).lower()})
                if author_elem:
                    author_name = author_elem.get_text().strip()
                    author_name = re.sub(r'^(By|Author|Written by|Posted by):?\s*', '', author_name, flags=re.IGNORECASE)
                    if author_name and len(author_name) < 100: 
                        break
        
        if not author_name:
            text_content = soup.get_text()
            name_pattern = r'([A-Z][a-z]+ [A-Z][a-z]+)\s+\1'
            match = re.search(name_pattern, text_content)
            if match:
                author_name = match.group(1)
        
        if not pub_year:
            date_metas = [
                soup.find('meta', attrs={'property': 'article:published_time'}),
                soup.find('meta', attrs={'name': 'publication_date'}),
                soup.find('meta', attrs={'name': 'date'})
            ]
            for meta in date_metas:
                if meta and meta.get('content'):
                    date_str = meta.get('content')
                    year_match = re.search(r'(20\d{2})', date_str)
                    if year_match:
                        pub_year = year_match.group(1)
                        break
        
        if not pub_year:
            text_content = soup.get_text()
            date_patterns = [
                r'Updated on:?\s*[A-Za-z]+\s+\d+,\s+(20\d{2})',
                r'Published:?\s*[A-Za-z]+\s+\d+,\s+(20\d{2})',
                r'©\s*(20\d{2})',
                r'Copyright\s*(20\d{2})'
            ]
            for pattern in date_patterns:
                match = re.search(pattern, text_content)
                if match:
                    pub_year = match.group(1)
                    break
        
        formatted_author = None
        if author_name:
            author_name = author_name.strip()
            author_name = re.sub(r'\s+', ' ', author_name)
            
            name_parts = author_name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name_initial = name_parts[-1][0] if name_parts[-1] else ''
                formatted_author = f"{first_name} {last_name_initial}."
            elif len(name_parts) == 1:
                formatted_author = name_parts[0]
        
        organization = None
        if '://' in url:
            domain = url.split('/')[2]
            org_parts = domain.replace('www.', '').split('.')
            organization = org_parts[0].capitalize() if org_parts else None
        
        return {
            'author': formatted_author,
            'year': pub_year or 'n.d.',
            'organization': organization
        }
        
    except requests.Timeout:
        logger.warning(f"Timeout while fetching web author from: {url[:80]}")
        return {'author': None, 'year': 'n.d.', 'organization': None}
    except Exception as e:
        logger.error(f"Failed to extract web author from {url[:80]}: {e}")
        return {'author': None, 'year': 'n.d.', 'organization': None}

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("pytesseract not available - OCR disabled")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


def download_pdf_with_full_metadata(pdf_url: str, max_pages: int = 5) -> Dict[str, any]:
    """
    Download PDF and extract BOTH text and images for OCR
    Gets more pages to ensure we capture author info
    
    Args:
        pdf_url: Full URL to PDF
        max_pages: Number of pages to extract (default 5 for thorough metadata)
    
    Returns:
        {
            'text': 'extracted text',
            'images': [PIL Images],
            'metadata': {} PDF metadata dict,
            'first_page_text': 'text from page 1 only'
        }
    """
    try:
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()
        
        pdf_bytes = io.BytesIO(response.content)
        result = {
            'text': '',
            'images': [],
            'metadata': {},
            'first_page_text': ''
        }
        
        with pdfplumber.open(pdf_bytes) as pdf:
            # Extract PDF metadata (often has author)
            if pdf.metadata:
                result['metadata'] = pdf.metadata
                logger.info(f"PDF metadata: {pdf.metadata}")
            
            # Extract text from first N pages
            text_parts = []
            for page_num in range(min(max_pages, len(pdf.pages))):
                page = pdf.pages[page_num]
                page_text = page.extract_text() or ''
                
                # Store first page separately for focused analysis
                if page_num == 0:
                    result['first_page_text'] = page_text
                
                text_parts.append(page_text)
                
                # Extract images for OCR (in case it's scanned)
                if TESSERACT_AVAILABLE and page_num == 0:
                    try:
                        images = page.images
                        for img_info in images[:3]:
                            bbox = (img_info['x0'], img_info['top'], 
                                   img_info['x1'], img_info['bottom'])
                            img = page.within_bbox(bbox).to_image()
                            if img:
                                result['images'].append(img.original)
                    except Exception as e:
                        logger.debug(f"Could not extract images from page {page_num}: {e}")
            
            result['text'] = '\n'.join(text_parts)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to download/extract PDF: {e}")
        return {'text': '', 'images': [], 'metadata': {}, 'first_page_text': ''}


async def extract_author_with_ai(first_page_text: str) -> Optional[str]:
    """
    Use Claude API to intelligently extract author names from PDF text
    This handles ANY format including edge cases regex can't catch
    
    Args:
        first_page_text: First page of PDF as text
        
    Returns:
        First author's last name in "LastName et al." format, or None
    """
    if not first_page_text or len(first_page_text) < 50:
        return None
    
    # Check if ANTHROPIC_API_KEY is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set - skipping AI extraction")
        return None
    
    try:
        # Take first 2000 chars for API efficiency
        text_sample = first_page_text[:2000]
        
        # Create prompt for Claude
        prompt = f"""Extract the FIRST author's last name from this academic paper's first page.

Rules:
- Return ONLY the last name of the FIRST author (not all authors)
- Ignore title, abstract, keywords, affiliations, emails
- Ignore journal names, copyright notices, headers
- If multiple authors are listed, take ONLY the first one
- Return format: just the last name (e.g., "Smith" not "John Smith")
- If you cannot find a clear author name, return: NONE

First page text:
{text_sample}

FIRST AUTHOR'S LAST NAME:"""
        
        # Call Claude API
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            author_name = data['content'][0]['text'].strip()
            
            # Validate response
            if author_name and author_name.upper() != 'NONE' and len(author_name) < 50:
                # Clean up any extra text
                author_name = re.sub(r'[.,;:\*\'"()]', '', author_name).strip()
                
                # Make sure it's a valid name (letters only, possibly with dash)
                if re.match(r'^[A-Za-z\-]+$', author_name):
                    # Capitalize properly
                    author_name = author_name.capitalize() if author_name.islower() else author_name
                    logger.info(f"✓ AI extracted author: {author_name}")
                    return f"{author_name} et al."
        
        logger.warning("AI extraction failed or returned invalid result")
        return None
        
    except Exception as e:
        logger.error(f"AI author extraction failed: {e}")
        return None


def extract_author_with_ocr_fallback(pdf_data: Dict, filename: str) -> Dict[str, str]:
    """
    COMPREHENSIVE author extraction with multiple strategies:
    1. PDF metadata (most reliable)
    2. AI-powered extraction using Claude API (NEW - handles ANY format)
    3. Regex-based extraction (fallback for when AI unavailable)
    4. OCR on images (for scanned PDFs)
    5. Filename parsing (last resort)
    
    Returns author in "LastName et al." format for APA
    
    Args:
        pdf_data: Output from download_pdf_with_full_metadata()
        filename: Fallback filename
        
    Returns:
        {'author': 'LastName et al.', 'year': 'YYYY', 'title': 'Paper Title'}
    """
    result = {'author': None, 'year': None, 'title': None}
    
    # STRATEGY 1: PDF Metadata (most reliable)
    metadata = pdf_data.get('metadata', {})
    if metadata:
        author_fields = ['Author', 'author', 'Creator', 'creator', 'Authors', 'authors']
        for field in author_fields:
            if field in metadata and metadata[field]:
                author_raw = str(metadata[field]).strip()
                author_raw = re.sub(r'^(by|author:?)\s*', '', author_raw, flags=re.IGNORECASE)
                
                if author_raw and len(author_raw) < 200:
                    result['author'] = format_author_for_apa(author_raw)
                    logger.info(f"✓ Author from PDF metadata: {result['author']}")
                    break
        
        # Check for year in metadata
        date_fields = ['CreationDate', 'creation_date', 'ModDate', 'mod_date', 'Date', 'date']
        for field in date_fields:
            if field in metadata and metadata[field]:
                year_match = re.search(r'(19\d{2}|20\d{2})', str(metadata[field]))
                if year_match:
                    result['year'] = year_match.group(1)
                    break
    
    # STRATEGY 2: AI-powered extraction (NEW - most reliable for text)
    if not result['author']:
        first_page = pdf_data.get('first_page_text', '')
        if first_page:
            # Try AI extraction first
            try:
                import asyncio
                # Check if we're already in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in async context, create task
                    ai_author = loop.run_until_complete(extract_author_with_ai(first_page))
                except RuntimeError:
                    # Not in async context, create new loop
                    ai_author = asyncio.run(extract_author_with_ai(first_page))
                
                if ai_author:
                    result['author'] = ai_author
                    logger.info(f"✓ Author from AI: {result['author']}")
            except Exception as e:
                logger.warning(f"AI extraction error: {e}")
    
    # STRATEGY 3: Regex-based extraction (fallback)
    if not result['author']:
        first_page = pdf_data.get('first_page_text', '')
        if first_page:
            author = extract_author_with_regex(first_page)
            if author:
                result['author'] = author
                logger.info(f"✓ Author from regex: {result['author']}")
    
    # Extract year from text if not found
    text = pdf_data.get('text', '')
    if text and not result['year']:
        header = text[:2000]
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', header)
        if year_matches:
            years = [int(y) for y in year_matches if 1990 <= int(y) <= 2025]
            if years:
                result['year'] = str(max(years))
    
    # STRATEGY 4: OCR on images (for scanned PDFs)
    if not result['author'] and TESSERACT_AVAILABLE and pdf_data.get('images'):
        logger.info("Attempting OCR on PDF images...")
        ocr_author = extract_author_from_images(pdf_data['images'])
        if ocr_author:
            result['author'] = ocr_author
            logger.info(f"✓ Author from OCR: {result['author']}")
    
    # STRATEGY 5: Filename parsing (last resort)
    if not result['author'] and filename:
        filename_data = extract_author_year_from_filename(filename)
        if filename_data.get('author'):
            result['author'] = filename_data['author']
            logger.info(f"⚠ Author from filename: {result['author']}")
        if filename_data.get('year') and not result['year']:
            result['year'] = filename_data['year']
    
    # Extract title
    if filename:
        title = filename.replace('.pdf', '').replace('.json', '')
        title = re.sub(r'^paper_\d+_?', '', title)
        title = title.replace('_', ' ').strip()
        result['title'] = title[:200]
    
    # Final defaults
    if not result['author']:
        result['author'] = 'Unknown Author'
        logger.warning(f"❌ Could not extract author from PDF: {filename}")
    if not result['year']:
        result['year'] = 'n.d.'
    
    return result


def extract_author_with_regex(first_page_text: str) -> Optional[str]:
    """
    Regex-based author extraction (fallback when AI unavailable)
    Handles common patterns but won't catch all edge cases
    """
    if not first_page_text or len(first_page_text) < 50:
        return None
    
    text = first_page_text[:2000]
    
    # Common false positives to avoid
    false_positives = {
        'abstract', 'introduction', 'keywords', 'university', 'college',
        'journal', 'copyright', 'published', 'supervision', 'implementation',
        'system', 'version', 'equation', 'windows', 'unicode'
    }
    
    def is_false_positive(name: str) -> bool:
        name_lower = name.lower()
        return any(fp in name_lower for fp in false_positives)
    
    def extract_last_name(name: str) -> Optional[str]:
        """Extract last name from full name"""
        # Clean credentials
        name = re.sub(r',?\s*(?:Ph\.?D\.?|M\.D\.?|M\.P\.P\.?|M\.A\.?|M\.S\.?|EdD|MD|JD|Psy\.?D\.?).*$', 
                     '', name, flags=re.IGNORECASE)
        parts = name.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            last_name = re.sub(r'[,;:\*\.]', '', last_name)
            if last_name.isupper() and len(last_name) > 2:
                last_name = last_name.capitalize()
            return f"{last_name} et al."
        return None
    
    # Pattern 1: Multiple authors with middle initials
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)+\s+[A-Z][a-z]+)'
    matches = re.findall(pattern, text[:500])  # First 500 chars most likely
    for match in matches:
        if not is_false_positive(match):
            result = extract_last_name(match)
            if result:
                return result
    
    # Pattern 2: Simple "FirstName LastName"
    pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
    matches = re.findall(pattern, text[:500])
    for match in matches:
        if not is_false_positive(match) and len(match.split()) == 2:
            result = extract_last_name(match)
            if result:
                return result
    
    return None


def extract_author_from_images(images: list) -> Optional[str]:
    """Use OCR to extract author from PDF images"""
    if not TESSERACT_AVAILABLE:
        return None
    
    try:
        for img in images[:2]:
            img_gray = img.convert('L')
            ocr_text = pytesseract.image_to_string(img_gray)
            
            if ocr_text:
                # Try AI extraction on OCR text
                try:
                    import asyncio
                    ai_author = asyncio.run(extract_author_with_ai(ocr_text))
                    if ai_author:
                        return ai_author
                except:
                    pass
                
                # Fallback to regex
                author = extract_author_with_regex(ocr_text)
                if author:
                    return author
        
        return None
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return None


def format_author_for_apa(author_raw: str) -> str:
    """Format author name for APA style: "LastName et al." """
    if not author_raw:
        return None
    
    author_raw = author_raw.strip()
    author_raw = re.sub(r'\s+', ' ', author_raw)
    author_raw = re.sub(r',\s*(?:Ph\.?D\.?|M\.D\.?|M\.P\.P\.?|M\.A\.?|M\.S\.?|EdD|MD|JD|Psy\.?D\.?).*$', 
                       '', author_raw, flags=re.IGNORECASE)
    
    if ',' in author_raw:
        last_name = author_raw.split(',')[0].strip()
        if last_name.isupper() and len(last_name) > 2:
            last_name = last_name.capitalize()
        return f"{last_name} et al."
    
    parts = author_raw.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        last_name = re.sub(r'[,;:\*\.]', '', last_name)
        if last_name.isupper() and len(last_name) > 2:
            last_name = last_name.capitalize()
        return f"{last_name} et al."
    
    name = author_raw.capitalize() if author_raw.isupper() else author_raw
    return f"{name} et al."


def extract_author_year_from_filename(filename: str) -> Dict[str, str]:
    """Extract author and year from filename (fallback)"""
    result = {'author': None, 'year': None}
    
    if not filename:
        return result
    
    name = filename.replace('.pdf', '').replace('.json', '')
    
    match = re.search(r'([A-Z][a-z]+)(?:\s+et\s+al\.?)?\s*\((\d{4})\)', name)
    if match:
        result['author'] = f"{match.group(1)} et al."
        result['year'] = match.group(2)
        return result
    
    match = re.search(r'^(\d{4})[_\-\s]+([A-Z][a-z]+)', name)
    if match:
        result['year'] = match.group(1)
        result['author'] = f"{match.group(2)} et al."
        return result
    
    match = re.search(r'([A-Z][a-z]+)[_\-\s]+(\d{4})', name)
    if match:
        result['author'] = f"{match.group(1)} et al."
        result['year'] = match.group(2)
        return result
    
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', name)
    if year_match:
        result['year'] = year_match.group(1)
    
    return result

def extract_metadata_from_content(content: str, filename: str) -> Dict[str, str]:
    """
    PRODUCTION-READY author extraction - ACTUALLY reads PDF content
    Based on proven working extractor
    
    Handles:
    - ALL CAPS with asterisks: "JOANNA KAMYKOWSKA*, EWA HAMAN**"
    - Mixed case: "Stephen Basil Scott, Kathy Sylva"  
    - Single names: "Josephine Nartey"
    """
    result = {'author': None, 'year': None, 'title': None}
    
    if not content:
        return result
    
    header = content[:8000]
    
    lines = [line.strip() for line in header.split('\n') if line.strip()]
    
    if len(lines) < 5:
        lines = re.split(r'(?<=[.!?])\s+(?=[A-Z])|  +', header)
        lines = [line.strip() for line in lines if line.strip() and len(line) > 5]
        
    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', header)
    if year_matches:
        years = [int(y) for y in year_matches if 1990 <= int(y) <= 2025]
        if years:
            result['year'] = str(max(years))
    
    false_positives = {
        'Abstract', 'Introduction', 'Method', 'Results', 'Discussion', 'Conclusion',
        'References', 'Appendix', 'Table', 'Figure', 'Keywords', 'Acknowledgments',
        'University', 'College', 'Institute', 'Department', 'School', 'Center', 'Centre',
        'Developmental', 'Brief', 'Career', 'Papers', 'From', 'Research', 'Study',
        'Analysis', 'Report', 'Review', 'Journal', 'Volume', 'Issue', 'Page', 'Copyright',
        'Metric', 'Equation', 'Earnings', 'Jobs', 'Eviction', 'Prevention', 'Decreased',
        'Juvenile', 'Delinquency', 'Parents', 'Likely', 'Work', 'Child', 'Care',
        'Intensive', 'Fostering', 'Independent', 'Evaluation', 'English', 'Setting',
        'Pilot', 'Trial', 'Supporting', 'Teens', 'Academic', 'Needs', 'Daily', 'Stand',
        'Parent', 'Adolescent', 'Collaborative', 'Intervention', 'Adhd', 'Sustainability',
        'Effects', 'Multisystemic', 'Therapy', 'Netherlands', 'Randomized', 'Controlled',
        'Adjunctive', 'Family', 'Treatment', 'Usual', 'Following', 'Inpatient', 'Anorexia',
        'Nervosa', 'Adolescents', 'Preventing', 'Falls', 'Physically', 'Active', 'Community',
        'Dwelling', 'Older', 'People', 'Comparison', 'Two', 'Techniques', 'Instructions'
    }
    
    def is_valid_author_name(name: str) -> bool:
        """Check if a name is likely a real author name, not a false positive"""
        parts = name.split()
        for part in parts:
            if part.capitalize() in false_positives or part.lower() in ['the', 'a', 'an', 'and', 'or', 'of', 'for', 'to', 'in', 'on']:
                return False
        if len(parts) < 2:
            return False
        if any(char in name for char in [',', ';', ':', '(', ')', '[', ']']):
            return False
        return True
    
    for i, line in enumerate(lines[:100]):
        if not line or len(line) > 200:
            continue
        
        caps_asterisk = re.findall(r'\b([A-Z][A-Z\-\s]{3,40}?[A-Z])\*+', line)
        if caps_asterisk:
            name = caps_asterisk[0].strip()
            name = re.sub(r'\s+', ' ', name)
            if len(name) > 3 and is_valid_author_name(name):
                parts = name.split()
                if parts:
                    last = parts[-1].capitalize()
                    result['author'] = f"{last} et al."
                    break
        
        mixed = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', line)
        if mixed:
            name = mixed[0].strip()
            if is_valid_author_name(name):
                parts = name.split()
                if len(parts) >= 2:
                    last = parts[-1]
                    result['author'] = f"{last} et al."
                    break
        
        if re.search(r'\bauthor[s]?\s*[:]\s*', line, re.IGNORECASE):
            section = '\n'.join(lines[i:i+5])
            names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b', section)
            if names:
                for name in names:
                    if is_valid_author_name(name):
                        parts = name.split()
                        if len(parts) >= 2:
                            last = parts[-1]
                            result['author'] = f"{last} et al."
                            break
                if result['author']:
                    break
        
        if result['author']:
            break
    
    if filename:
        title = filename.replace('.pdf', '').replace('.json', '')
        title = re.sub(r'^paper_\d+_?', '', title)
        title = title.replace('_', ' ').strip()
        result['title'] = title[:200] if len(title) > 200 else title
    
    if not result['author']:
        logger.warning(f"Could not extract author from PDF content (filename: {filename})")
    
    return result


def normalize_filename(filename: str) -> str:
    """
    Normalize filename for strict deduplication
    Removes (1), (2), etc. suffixes, extensions, and normalizes case
    
    Example: "Report (1).pdf" → "report"
    """
    name = filename.replace('.pdf', '').replace('.json', '')
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    name = re.sub(r'^paper_\d+_?', '', name)
    name = name.strip().lower()
    return name


async def search_citation_metadata(filename: str, paper_id: str) -> Dict[str, str]:
    """
    Search for paper metadata (author, year) using Tavily web search.
    
    Args:
        filename: The paper filename
        paper_id: The paper ID
        
    Returns:
        Dict with 'author' and 'year' keys
    """
    result = {'author': 'Unknown Author', 'year': 'n.d.'}
    
    if not tavily_client:
        return result
    
    try:
        clean_name = filename.replace('.pdf', '').replace('.json', '').replace('_', ' ')
        
        if len(clean_name) > 150:
            clean_name = clean_name[:150].rsplit(' ', 1)[0]  
        
        search_query = clean_name[:400]  
                
        response = tavily_client.search(
            query=search_query,
            max_results=2,
            search_depth="basic"
        )
        
        if response and 'results' in response and response['results']:
            for item in response['results']:
                content = item.get('content', '') + ' ' + item.get('title', '')
                
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', content)
                if year_match and result['year'] == 'n.d.':
                    result['year'] = year_match.group(1)
                
                if result['author'] == 'Unknown Author':
                    author_match = re.search(r'([A-Z][a-z]+),\s*([A-Z]\.(?:\s*[A-Z]\.)?)', content)
                    if author_match:
                        result['author'] = f"{author_match.group(1)} {author_match.group(2)}"
                        continue
                    
                    author_match = re.search(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+et\s+al', content)
                    if author_match:
                        result['author'] = f"{author_match.group(1)} et al."
                        continue
                    
                    author_match = re.search(r'([A-Z][a-z]+)\s+et\s+al', content)
                    if author_match:
                        result['author'] = f"{author_match.group(1)} et al."            
        else:
            logger.info(f"No results found for citation metadata search")
        
    except Exception as e:
        logger.error(f"Citation metadata search failed: {e}")
    
    return result


def get_pdf_url_from_paper_id(paper_id: str) -> Optional[str]:
    """
    Generate PDF URL from paper_id using Supabase Storage
    
    First checks if the paper exists in research_corpus (uploaded files),
    then generates the appropriate URL.
    
    Args:
        paper_id: The paper identifier from research_chunks
        
    Returns:
        Full URL to PDF or None if not available
    """
    if not paper_id or paper_id == 'N/A':
        return None
    
    if paper_id.endswith('.json'):
        return None
    
    try:
        from app.services.supabase_client import get_storage_filename_from_paper_id
        
        storage_filename = get_storage_filename_from_paper_id(paper_id)
        
        if not storage_filename:
            return None
            
    except Exception as e:
        logger.error(f"[PDF URL] Could not verify paper_id in storage: {e}")
        storage_filename = f"{paper_id}.pdf"
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_KEY')
    bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'research-pdfs')
    
    if not supabase_url:
        logger.error("[PDF URL] SUPABASE_URL not configured")
        return None
    
    supabase_url = supabase_url.rstrip('/')
    
    use_signed_urls = os.getenv('USE_SIGNED_PDF_URLS', 'false').lower() == 'true'
    
    if not use_signed_urls:
        return f"{supabase_url}/storage/v1/object/public/{bucket_name}/{storage_filename}"
    
    if not supabase_key:
        logger.error("[PDF URL] SUPABASE_KEY required for signed URLs")
        return f"{supabase_url}/storage/v1/object/public/{bucket_name}/{paper_id}.pdf"
    
    try:
        from supabase import create_client
        
        supabase = create_client(supabase_url, supabase_key)
        
        result = supabase.storage.from_(bucket_name).create_signed_url(
            f"{paper_id}.pdf",
            expires_in=3600
        )
        
        if result and 'signedURL' in result:
            return result['signedURL']
        elif result and 'signed_url' in result:
            return result['signed_url']
        else:
            return None
            
    except ImportError:
        logger.error("[PDF URL] Supabase client not installed. Run: pip install supabase")
        return f"{supabase_url}/storage/v1/object/public/{bucket_name}/{paper_id}.pdf"
    except Exception as e:
        logger.error(f"[PDF URL] Failed to generate signed URL for {paper_id}: {e}")
        return f"{supabase_url}/storage/v1/object/public/{bucket_name}/{paper_id}.pdf"


class CitationManager:
    """Manages citations in APA format with author/year extraction"""
    
    def __init__(self):
        self.citations: List[Dict[str, Any]] = []
        self.citation_map: Dict[str, int] = {}  
        self.next_id = 1
        self.metadata_cache: Dict[str, Dict[str, str]] = {}
    
    async def add_citation(self, chunk: Dict[str, Any], source_type: str = "research") -> int:
        """
        Add a citation and return its ID (deduplicated)
        Supports both research corpus chunks and web search results
        
        Args:
            chunk: Citation data (from RAG or web search)
            source_type: "research" or "web"
        """
        if source_type == "research":
            paper_id = chunk.get('paper_id', f"chunk_{self.next_id}")
            filename = chunk.get('filename', 'N/A')
            
            normalized_filename = normalize_filename(filename)
            unique_key = f"research_{paper_id}_{normalized_filename}"
            
        else:  
            url = chunk.get('url', '')
            url = url.rstrip('/')
            unique_key = f"web_{url}"
        
        if unique_key in self.citation_map:
            logger.info(f"Duplicate citation detected, skipping: {unique_key}")
            return self.citation_map[unique_key]
        
        citation_id = self.next_id
        self.next_id += 1
        
        self.citation_map[unique_key] = citation_id
        
        if source_type == "research":
            paper_id = chunk.get('paper_id', 'N/A')
            filename = chunk.get('filename', 'N/A')
            content = chunk.get('content', '')
            
            pdf_url = chunk.get('pdf_url')
            if not pdf_url and paper_id and paper_id != 'N/A':
                pdf_url = get_pdf_url_from_paper_id(paper_id)
            
            if paper_id not in self.metadata_cache:
                pdf_url = chunk.get('pdf_url')
                if not pdf_url and paper_id and paper_id != 'N/A':
                    pdf_url = get_pdf_url_from_paper_id(paper_id)
                
                # NEW: Use comprehensive extraction
                if pdf_url:
                    logger.info(f"Extracting metadata from PDF: {pdf_url}")
                    pdf_data = download_pdf_with_full_metadata(pdf_url, max_pages=5)
                    metadata = extract_author_with_ocr_fallback(pdf_data, filename)
                else:
                    # Fallback to filename only
                    logger.warning(f"No PDF URL available for {paper_id}, using filename only")
                    metadata = extract_author_year_from_filename(filename)
                    if not metadata['author']:
                        metadata['author'] = 'Unknown Author'
                    if not metadata['year']:
                        metadata['year'] = 'n.d.'
                
                self.metadata_cache[paper_id] = metadata
            else:
                metadata = self.metadata_cache[paper_id]
            
            if metadata.get('author') and metadata.get('author') != 'Unknown Author' and metadata.get('year'):
                author_year_key = f"research_{metadata['author']}_{metadata['year']}"
                if author_year_key in self.citation_map:
                    existing_id = self.citation_map[author_year_key]
                    logger.info(f"Duplicate citation (author+year): {metadata['author']}, {metadata['year']}")
                    return existing_id
                self.citation_map[author_year_key] = citation_id
            
            pdf_url = chunk.get('pdf_url')
            if not pdf_url and paper_id and paper_id != 'N/A':
                pdf_url = get_pdf_url_from_paper_id(paper_id)
                logger.info(f"Generated PDF URL for {paper_id}: {pdf_url}")
            
            self.citations.append({
                'id': citation_id,
                'source_type': 'research',
                'chunk_id': chunk.get('chunk_id', 'N/A'),
                'paper_id': paper_id,
                'filename': filename,
                'author': metadata['author'],
                'year': metadata['year'],
                'section': chunk.get('section', 'N/A'),
                'domain': chunk.get('domain', 'N/A'),
                'content': chunk.get('content', '')[:300],
                'distance': float(chunk.get('distance', 0.0)),
                'pdf_url': pdf_url
            })
        else:  
            self.citations.append({
                'id': citation_id,
                'source_type': 'web',
                'chunk_id': unique_key,
                'paper_id': 'web',
                'filename': chunk.get('title', 'Web Source'),
                'author': chunk.get('author', 'Unknown'),
                'year': chunk.get('year', 'n.d.'),
                'section': 'Web',
                'domain': chunk.get('domain', 'web'),
                'organization': chunk.get('organization', 'Unknown'),
                'content': chunk.get('snippet', '')[:300],
                'distance': 0.0,
                'url': chunk.get('url', '')
            })
        
        return citation_id
    
    def format_reference_section(self) -> str:
        """Generate APA 7 style reference section (alphabetical, no numbers)"""
        if not self.citations:
            return ""
        
        lines = ["\n\n## References\n"]
        
        # Sort by author name (APA 7 style)
        sorted_citations = sorted(
            self.citations, 
            key=lambda x: (x.get('author', 'ZZZ').lower(), x.get('year', '9999'))
        )
        
        for citation in sorted_citations:
            if citation['source_type'] == 'research':
                author = citation.get('author', '')
                year = citation.get('year', 'n.d.')
                filename = citation['filename']
                pdf_url = citation.get('pdf_url', '')
                
                title = filename.replace('.pdf', '').replace('.json', '')
                title = re.sub(r'^paper_\d+_?', '', title)
                title = title.replace('_', ' ').strip()
                
                title = re.sub(r'^[A-Z][a-z]+\s+et\s+al\.?\s+\(\d{4}\)\s*', '', title)
                title = re.sub(r'^\d{4}\.?\s*', '', title)
                
                if len(title) > 200:
                    title = title[:197].rsplit(' ', 1)[0] + "..."
                
                if author and author != 'Unknown Author':
                    if pdf_url:
                        ref_line = f"- {author} ({year}). *{title}*. Retrieved from {pdf_url}"
                    else:
                        ref_line = f"- {author} ({year}). *{title}*."
                else:
                    if pdf_url:
                        ref_line = f"- *{title}*. ({year}). Retrieved from {pdf_url}"
                    else:
                        ref_line = f"- *{title}*. ({year})."
                
                lines.append(ref_line)
            else:  
                source_title = citation.get('filename', 'Web Source')
                url = citation.get('url', 'N/A')
                author = citation.get('author', '')
                year = citation.get('year', 'n.d.')
                
                clean_title = source_title.replace('|', '-').strip()
                clean_title = re.sub(r'\b20\d{2}\b', '', clean_title).strip()
                clean_title = re.sub(r'\s+', ' ', clean_title)
                clean_title = re.sub(r'\[PDF\]\s*', '', clean_title, flags=re.IGNORECASE)
                clean_title = re.sub(r'\[XML\]\s*', '', clean_title, flags=re.IGNORECASE)
                clean_title = clean_title.strip()
                
                if author and author != 'Unknown':
                    ref_line = f"- {author}. ({year}). *{clean_title}*. Retrieved from {url}"
                else:
                    ref_line = f"- *{clean_title}*. ({year}). Retrieved from {url}"
                
                lines.append(ref_line)
        
        return "\n".join(lines)
    
    def get_citations_metadata(self) -> List[Dict[str, Any]]:
        """Return citation metadata for storage/display"""
        return self.citations


async def search_web_for_context(
    query: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Web search fallback using Tavily Python client when RAG returns no results
    Returns authoritative sources with verifiable URLs for citations
    
    Args:
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of web results with title, snippet, url, domain
    """
    logger.info(f"Initiating Tavily web search for query: '{query[:100]}...'")
    
    try:
        from tavily import TavilyClient
        
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        
        if not tavily_api_key:
            logger.error("TAVILY_API_KEY not found in environment - web search disabled")
            return []
        
        client = TavilyClient(api_key=tavily_api_key)
        
        truncated_query = query[:400] if len(query) > 400 else query
        if len(query) > 400:
            logger.debug(f"Query truncated from {len(query)} to 400 characters")
                
        response = client.search(
            query=truncated_query,
            search_depth="advanced",  
            max_results=min(top_k, 10)  
        )
        
        results = response.get('results', [])
        
        logger.info(f"Tavily returned {len(results)} results")
        
        web_results = []
        for result in results:
            url = result.get('url', '')
            title = result.get('title', 'Untitled')
            
            author_data = extract_author_from_webpage(url, title)
            
            author_name = author_data.get('author')
            pub_year = author_data.get('year', 'n.d.')
            
            if not author_name:
                if '://' in url:
                    domain = url.split('/')[2]
                    org_parts = domain.replace('www.', '').replace('www3.', '').split('.')
                    
                    if 'gov' in org_parts:
                        if 'erie' in domain:
                            org_name = 'Erie County'
                        elif 'santamonica' in domain:
                            org_name = 'City of Santa Monica'
                        else:
                            org_name = org_parts[0].replace('-', ' ').title()
                    elif '.org' in domain or '.edu' in domain:
                        org_name = org_parts[0].replace('-', ' ').title()
                    else:
                        org_name = org_parts[0].capitalize()
                    
                    if url.endswith('.pdf') and title:
                        if ' - ' in title:
                            potential_org = title.split(' - ')[-1].strip()
                            if len(potential_org) < 80 and any(word in potential_org.lower() for word in ['county', 'city', 'department', 'district', 'commission']):
                                org_name = potential_org
                        org_match = re.search(r'((?:County|City|Town|Department|District|Commission|Office)\s+of\s+[A-Z][a-zA-Z\s]+)', title, re.IGNORECASE)
                        if org_match:
                            org_name = org_match.group(1).strip()
                    
                    org_name = re.sub(r'\s+', ' ', org_name).strip()
                    if not org_name or org_name.lower() in ['www', 'www3', 'cdn', 'static']:
                        org_name = 'Unknown Source'
                else:
                    org_name = 'Unknown Source'
                
                author_name = org_name
            else:
                logger.debug(f"Extracted web author: {author_name}, year: {pub_year}")
            
            domain = url.split('/')[2] if '://' in url else 'unknown'
            
            web_results.append({
                'title': title,
                'snippet': result.get('content', '')[:500],  
                'url': url,
                'domain': domain,
                'author': author_name, 
                'year': pub_year,
                'organization': author_data.get('organization'),  
                'score': result.get('score', 0.0)  
            })
        
        logger.info(f"Successfully processed {len(web_results)} web results for citations")
        return web_results
        
    except ImportError:
        logger.error("Tavily client not installed. Run: pip install tavily-python")
        return []
    except Exception as e:
        logger.error(f"Tavily web search failed: {e}")
        import traceback
        traceback.print_exc()
        return []


async def search_rag_with_web_fallback(
    user_query: str,
    top_k: int = 10,
    domain: Optional[str] = None,
    enable_web_fallback: bool = False,
    always_include_web: bool = False
) -> Dict[str, Any]:
    """
    Search RAG system with optional web search as supplementary context
    
    Args:
        user_query: User's question or analysis request
        top_k: Number of research chunks to retrieve
        domain: Optional domain filter
        enable_web_fallback: Whether to use web search if RAG returns nothing
        always_include_web: Whether to always include web results alongside RAG (for real-time data)
    
    Returns:
        {
            'context': 'formatted context for prompt',
            'citations': CitationManager instance,
            'chunks': [raw chunks],
            'source': 'research' or 'web' or 'mixed'
        }
    """
    
    logger.info(f"Starting RAG search for query: '{user_query[:100]}...'")
    logger.debug(f"Web mode: {'Always include' if always_include_web else 'Fallback only'}")
    
    citation_mgr = CitationManager()
    
    try:
        chunks = await search_research_chunks_from_text(
            query_text=user_query,
            top_k=top_k,
            domain=domain
        )
        logger.info(f"Retrieved {len(chunks)} research chunks from vector database")
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        chunks = []
    
    web_results = []
    
    # Only perform web search if explicitly enabled
    if enable_web_fallback:
        if always_include_web:
            web_results = await search_web_for_context(user_query, top_k=5)
            if web_results:
                logger.info(f"Retrieved {len(web_results)} supplementary web results")
            else:
                logger.warning("No supplementary web results found")
        elif not chunks:
            web_results = await search_web_for_context(user_query, top_k=5)
            if web_results:
                logger.info(f"Retrieved {len(web_results)} fallback web results")
            else:
                logger.warning("No fallback web results found")
    
    if not chunks and not web_results:
        logger.warning("No research or web results found, proceeding without citations")
        return {
            'context': '',
            'citations': citation_mgr,
            'chunks': [],
            'source': 'none'
        }
    
    context_lines = ["\n## Research Evidence\n"]
    
    if chunks:
        context_lines.append(
            "The following peer-reviewed research has been retrieved from the knowledge base. "
            "Use these sources to support your analysis and cite them using [number, Author, Year] notation.\n"
        )
        
        for chunk in chunks:
            cit_id = await citation_mgr.add_citation(chunk, source_type="research")
            content_preview = chunk.get('content', '')[:400].strip()
            
            citation_data = next((c for c in citation_mgr.citations if c['id'] == cit_id), None)
            author = citation_data.get('author') if citation_data else None
            year = citation_data.get('year') if citation_data else None
            
            citation_header = f"[{cit_id}"
            
            is_real_author = False
            if author and author != 'Unknown Author':
                if 'et al' in author.lower() or len(author.split()) > 1:
                    is_real_author = True
            
            if is_real_author:
                citation_header += f", {author}"
            if year and year != 'n.d.':
                citation_header += f", {year}"
            citation_header += "]"
            
            context_lines.append(
                f"\n{citation_header} {content_preview}{'...' if len(chunk.get('content', '')) > 400 else ''}"
            )
            context_lines.append(
                f"    (Source: {chunk.get('filename', 'N/A')}, Section: {chunk.get('section', 'N/A')}, "
                f"Domain: {chunk.get('domain', 'N/A')})"
            )
    
    if web_results:
        if chunks:
            context_lines.append("\n## Additional Web Sources\n")
        else:
            context_lines.append(
                "The following sources were found via web search. "
                "Use these to support your analysis and cite them using [number, Author, Year] notation.\n"
            )
        
        for result in web_results:
            cit_id = await citation_mgr.add_citation(result, source_type="web")
            snippet = result.get('snippet', '')[:400]
            
            citation_data = next((c for c in citation_mgr.citations if c['id'] == cit_id), None)
            author = citation_data.get('author') if citation_data else None
            year = citation_data.get('year', 'n.d.') if citation_data else 'n.d.'
            
            citation_header = f"[{cit_id}"
            if author:
                citation_header += f", {author}"
                if year:
                    citation_header += f", {year}"
            citation_header += "]"
            
            context_lines.append(f"\n{citation_header} {snippet}")
            context_lines.append(f"    (Web Source: {result.get('url', 'N/A')})")
    
    context_lines.append("\n---\n")
    
    source_type = "research" if chunks and not web_results else \
                  "web" if web_results and not chunks else \
                  "mixed"
        
    return {
        'context': '\n'.join(context_lines),
        'citations': citation_mgr,
        'chunks': chunks + web_results,
        'source': source_type
    }


async def generate_with_rag_citations(
    system_prompt: str,
    user_query: str,
    top_k_research: int = 10,
    domain: Optional[str] = None,
    max_tokens: int = 6000,
    enable_web_fallback: bool = False,
    always_include_web: bool = False
) -> Dict[str, Any]:
    """
    Generate response with automatic RAG retrieval, supplementary web search, and APA-style citations
    
    This is the main function to use in your route handlers.
    It handles RAG search, web search fallback, citation management, and response formatting.
    
    Args:
        system_prompt: Your existing system prompt (unchanged)
        user_query: User's question/request
        top_k_research: Number of research chunks to retrieve (default 10)
        domain: Optional domain filter ("education", "health", etc.)
        max_tokens: Max response length
        enable_web_fallback: Enable web search if RAG returns nothing
        always_include_web: Always include web search for real-time context (not just fallback)
    
    Returns:
        {
            'response': 'generated text with inline citations and APA references',
            'citations': [citation metadata],
            'has_research': bool,
            'num_sources': int,
            'source': 'research'|'web'|'mixed'|'none'
        }
    """
    
    logger.info(f"SINGLE ANALYSIS WITH RAG and SUPPLEMENTARY WEB SEARCH")
    
    rag_result = await search_rag_with_web_fallback(
        user_query=user_query,
        top_k=top_k_research,
        domain=domain,
        enable_web_fallback=enable_web_fallback,
        always_include_web=always_include_web
    )
    
    enhanced_system = system_prompt
    
    if rag_result['context']:
        enhanced_system = f"""{system_prompt}

{rag_result['context']}

## CITATION FORMAT REQUIREMENTS (APA 7 Inline Citations - MANDATORY)

**CRITICAL: Use APA 7 inline citation format throughout your response.**

**Required Format:**
- Single Author: (Smith, 2020)
- Two authors: (Smith & Johnson, 2020)
- Multiple authors: (Smith et al., 2020)
- Multiple sources: (Smith, 2020; Jones et al., 2019)
- Do NOT use numbered citations like [1], [2], [3]
- Only cite sources that have identifiable author names

**Placement (Strict):** Citations must appear **ONLY at the very end of the sentence**, immediately before the period.

**Correct Examples:**
✅ "The treatment group showed a 12% reduction in recidivism (Miller et al., 2015)."
✅ "Research demonstrates significant improvements in outcomes (Latessa et al., 2002)."
✅ "Multiple studies confirm program effectiveness (Smith, 2020; Miller et al., 2015)."

**Incorrect Examples:**
❌ "The study (Miller et al., 2015) showed a reduction..." (Mid-sentence placement)
❌ "Miller et al. (2015) states that the group showed..." (Narrative style)
❌ "Studies show impact [1]." (Numbered citations)

**Correction Strategy:** If you want to say "Miller found X", rephrase it to "Research shows X (Miller, 2015)."

**Important Rules:**
1. Extract author names from the research context provided
2. If no author name is available for a source, do NOT cite it inline
3. Use "n.d." only when year is explicitly unavailable
4. Match citations to actual sources in the context

**DO NOT generate a References section** - it will be automatically appended.

**Write a COMPREHENSIVE and DETAILED analysis** with specific data points and thorough examination.

Now proceed with your analysis using the evidence provided.
"""
    else:
        logger.debug("No evidence found, using base prompt")
    
    logger.info("Generating AI response with RAG context...")
    raw_response = safe_generate(
        system_msg=enhanced_system,
        user_msg=user_query,
        max_tokens=max_tokens
    )
    
    if not raw_response:
        logger.error(f"[RAG] Generation failed")
        return {
            'response': 'Sorry, I could not generate a response. Please try again.',
            'citations': [],
            'has_research': False,
            'num_sources': 0,
            'source': 'none'
        }
        
    final_response = raw_response
    
    if rag_result['citations'].citations:
        ref_section = rag_result['citations'].format_reference_section()
        final_response = raw_response + ref_section
        
    return {
        'response': final_response,
        'citations': rag_result['citations'].get_citations_metadata(),
        'has_research': len(rag_result['chunks']) > 0,
        'num_sources': len(rag_result['citations'].citations),
        'raw_chunks': rag_result['chunks'],
        'source': rag_result['source'] 
    }


def format_apa_citation_from_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into APA-style citation string.
    
    Args:
        metadata: Document metadata dict
    
    Returns:
        APA formatted citation string
    """
    # Check for explicit full citation first
    full_citation = (
        metadata.get("full_citation") or 
        metadata.get("Full Citation") or
        metadata.get("citation")
    )
    if full_citation:
        return full_citation
    
    parts = []
    
    # Authors
    authors = (
        metadata.get("authors") or 
        metadata.get("author") or 
        metadata.get("Author") or
        ""
    )
    if authors:
        parts.append(str(authors))
    
    # Year
    year = (
        metadata.get("year") or 
        metadata.get("Year") or 
        metadata.get("Year ") or
        "n.d."
    )
    parts.append(f"({year})")
    
    # Title
    title = (
        metadata.get("study_title") or 
        metadata.get("Study Title") or
        metadata.get("title") or
        metadata.get("filename") or
        ""
    )
    if title:
        title = title.replace(".pdf", "").replace("_", " ").strip()
        parts.append(title)
    
    # Source/Journal
    source = metadata.get("source") or metadata.get("journal") or ""
    if source:
        parts.append(source)
    
    return ". ".join([p for p in parts if p]) + "."


# =========================================================
# RELEVANCE CHECKING - DYNAMIC DETECTION
# =========================================================

# Minimum relevance score threshold (0.0 to 1.0)
# Documents below this threshold are considered irrelevant
MIN_RELEVANCE_SCORE = 0.65


def check_query_relevance(
    query_text: str,
    candidates: List[Dict[str, Any]],
    score_threshold: float = MIN_RELEVANCE_SCORE
) -> Dict[str, Any]:
    """
    Check if retrieved candidates are semantically relevant to the query.
    
    This prevents the system from forcing irrelevant citations when the
    database doesn't contain relevant data for the query topic.
    
    DYNAMIC DETECTION: Instead of hardcoding available domains, we extract
    topics/titles from the actual retrieved documents to show what was found.
    
    Args:
        query_text: The user's query or document content
        candidates: Retrieved candidate documents
        score_threshold: Minimum relevance score (0.0-1.0)
    
    Returns:
        Dict with:
            - is_relevant: bool
            - avg_score: float
            - max_score: float
            - relevant_count: int
            - detected_topics: Topics found in retrieved documents
            - recommendation: str (explanation for user)
    """
    if not candidates:
        return {
            "is_relevant": False,
            "avg_score": 0.0,
            "max_score": 0.0,
            "relevant_count": 0,
            "total_candidates": 0,
            "detected_topics": [],
            "detected_titles": [],
            "detected_meta_analyses": [],
            "recommendation": "No documents were retrieved from the database."
        }
    
    # Extract scores from candidates - handle 0 values explicitly
    scores = []
    for c in candidates:
        # Try different score fields - don't use 'or' which treats 0 as falsy
        score = c.get("rerank_score")
        if score is None:
            score = c.get("score")
        if score is None:
            score = c.get("distance")
        if score is None:
            score = 0.0
        
        # Normalize if needed (some scores are 0-1, some might be higher)
        if isinstance(score, (int, float)):
            scores.append(float(score))
    
    logger.debug(f"Relevance check scores: {scores[:5]}")
    
    if not scores:
        scores = [0.0]
    
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    
    # Count how many candidates meet the threshold
    relevant_count = sum(1 for s in scores if s >= score_threshold)
    
    # DYNAMIC: Extract topics/titles from the actual retrieved documents
    detected_topics = set()
    detected_titles = set()
    detected_meta_analyses = set()
    
    for c in candidates:
        md = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else c
        
        # Extract topic field
        topic = md.get("topic", "") or ""
        if topic and len(topic) > 10:
            detected_topics.add(topic[:80])
        
        # Extract study title
        study_title = md.get("study_title", "") or md.get("Study Title", "") or ""
        if study_title and len(study_title) > 10:
            detected_titles.add(study_title[:100])
        
        # Extract meta-analysis title (describes the category)
        meta_title = md.get("meta_analysis_title", "") or ""
        if meta_title and len(meta_title) > 10:
            detected_meta_analyses.add(meta_title[:100])
    
    # Determine if results are relevant
    # Criteria: At least 30% of results should meet threshold, OR max score > 0.75
    is_relevant = (relevant_count >= len(candidates) * 0.3) or (max_score >= 0.75)
    
    # Build recommendation message
    if is_relevant:
        recommendation = f"Found {relevant_count} relevant documents (avg relevance: {avg_score:.2f})"
    else:
        # Show what WAS found (dynamically detected)
        found_topics = list(detected_meta_analyses)[:3] or list(detected_topics)[:3]
        if found_topics:
            topics_str = "; ".join(found_topics)
            recommendation = (
                f"The retrieved documents have low relevance to your query "
                f"(avg score: {avg_score:.2f}). "
                f"The search found documents about: {topics_str}. "
                f"Consider refining your query or enabling web search."
            )
        else:
            recommendation = (
                f"The retrieved documents have low relevance to your query "
                f"(avg score: {avg_score:.2f}, max: {max_score:.2f}). "
                f"Consider refining your query or enabling web search."
            )
    
    return {
        "is_relevant": is_relevant,
        "avg_score": avg_score,
        "max_score": max_score,
        "relevant_count": relevant_count,
        "total_candidates": len(candidates),
        "detected_topics": list(detected_topics)[:10],
        "detected_titles": list(detected_titles)[:10],
        "detected_meta_analyses": list(detected_meta_analyses)[:10],
        "recommendation": recommendation
    }


def generate_no_relevant_data_response(
    query_text: str,
    relevance_check: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a helpful response when no relevant data is found.
    
    Instead of forcing irrelevant citations, this tells the user:
    1. That no relevant research was found for their topic
    2. What topics WERE found in the search results (dynamically detected)
    3. Suggestions for how to proceed
    
    Args:
        query_text: Original query
        relevance_check: Output from check_query_relevance()
    
    Returns:
        Dict with response and metadata
    """
    detected_topics = relevance_check.get("detected_topics", [])
    detected_titles = relevance_check.get("detected_titles", [])
    detected_meta_analyses = relevance_check.get("detected_meta_analyses", [])
    avg_score = relevance_check.get("avg_score", 0)
    max_score = relevance_check.get("max_score", 0)
    total_candidates = relevance_check.get("total_candidates", 0)
    
    # Build the response
    response = f"""## Research Relevance Notice

**The retrieved research does not appear to directly address your query.**

Rather than provide potentially misleading citations, I'm letting you know what was found and offering alternatives.

### Relevance Assessment

| Metric | Value | Threshold |
|--------|-------|-----------|
| Documents Retrieved | {total_candidates} | - |
| Average Relevance Score | {avg_score:.2f} | {MIN_RELEVANCE_SCORE} |
| Maximum Relevance Score | {max_score:.2f} | 0.75 |
| Status | **Below Threshold** | - |

"""
    
    # Show what WAS found (dynamically detected from results)
    if detected_meta_analyses:
        response += """### What the Search Found

The database search returned studies primarily in these research areas:

"""
        for meta in list(detected_meta_analyses)[:5]:
            response += f"- {meta}\n"
        response += "\n"
    
    if detected_titles:
        response += """### Sample Studies Retrieved

"""
        for title in list(detected_titles)[:3]:
            response += f"- *{title}*\n"
        response += "\n"
    
    if detected_topics:
        response += """### Topics in Retrieved Documents

"""
        for topic in list(detected_topics)[:5]:
            response += f"- {topic}\n"
        response += "\n"
    
    response += """### Why This Matters

The retrieved studies focus on different topics than your query. Citing them would be misleading because:
- The research context wouldn't support claims about your actual topic
- Readers might incorrectly assume the citations are relevant
- Analysis quality depends on using appropriate sources

### How to Proceed

**Option 1: Enable Web Search**
I can search the internet for relevant research on your topic. This may find studies that aren't in our curated database.

**Option 2: General Analysis**
I can analyze your document using my general knowledge without citing specific research from this database.

**Option 3: Refine Your Query**
If your document relates to the topics found above, try rephrasing your query to emphasize those connections.

**Option 4: Contact Administrator**
Request that relevant research for your topic area be added to the database.

---

Would you like me to proceed with one of these options?
"""
    
    return {
        "response": response,
        "citations": [],
        "has_research": False,
        "num_sources": 0,
        "raw_chunks": [],
        "source": "none",
        "relevance_check": relevance_check,
        "data_available": False
    }


def build_numbered_ref_context(
    candidates: List[Dict[str, Any]],
    max_content_chars: int = 2000
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build research context with author/year information for APA 7 citations.
    
    Format:
    ### SOURCE [1]
    AUTHORS: Smith et al.
    YEAR: 2020
    TITLE: Study Title
    FULL CITATION: Smith, J., Jones, K., & Brown, M. (2020). Title. Journal.
    LINK: http://...
    CONTENT EXCERPTS:
    text content here
    
    Args:
        candidates: List of candidate documents
        max_content_chars: Maximum characters for content excerpts
    
    Returns:
        Tuple of (formatted_context, citations_list)
    """
    context_parts = []
    citations = []
    
    for idx, c in enumerate(candidates, start=1):
        # Get metadata - handle both direct and nested metadata
        if isinstance(c.get("metadata"), dict):
            md = c.get("metadata", {})
        else:
            md = c  # Candidate itself is the metadata
        
        # Extract content
        content = (
            md.get("chunk_text") or 
            md.get("content") or 
            c.get("content") or
            md.get("text") or
            md.get("citation_text") or
            ""
        )[:max_content_chars]
        
        # Extract title
        title = (
            md.get("study_title") or 
            md.get("Study Title") or
            c.get("study_title") or
            md.get("filename") or
            c.get("filename") or
            "Untitled Document"
        )
        
        # Format APA citation
        citation = format_apa_citation_from_metadata({**md, **c})
        
        # Get link
        link = (
            md.get("study_link") or 
            md.get("gdrive_link") or
            md.get("Link to Full Study") or
            c.get("pdf_url") or
            md.get("link") or
            ""
        )
        
        # Extract authors for APA citation
        authors = (
            md.get("authors") or 
            md.get("author") or 
            md.get("Author") or
            c.get("author") or
            ""
        )
        
        # Extract year
        year = (
            md.get("year") or 
            md.get("Year") or 
            md.get("Year ") or
            c.get("year") or
            "n.d."
        )
        
        # Build context block with prominent author/year for APA citations
        block = (
            f"### SOURCE [{idx}]\n"
            f"AUTHORS: {authors}\n"
            f"YEAR: {year}\n"
            f"TITLE: {title}\n"
            f"FULL CITATION: {citation}\n"
            f"LINK: {link}\n"
            f"CONTENT EXCERPTS:\n{content}\n"
        )
        context_parts.append(block)
        
        # Store citation metadata for the citations array
        citations.append({
            "ref_id": idx,
            "id": c.get("id") or c.get("chunk_id"),
            "title": title,
            "full_citation": citation,
            "link": link,
            "score": c.get("rerank_score") or c.get("distance") or c.get("score", 0),
            "content_preview": content[:300],
            "filename": md.get("filename") or c.get("filename", ""),
            "domain": md.get("domain") or c.get("domain", ""),
            "section": md.get("section") or c.get("section", ""),
            "year": md.get("year") or c.get("year", "n.d."),
            "author": md.get("authors") or md.get("author") or c.get("author", "")
        })
    
    context = "\n\n".join(context_parts)
    logger.info(f"Built numbered context with {len(citations)} REF IDs")
    
    return context, citations


async def generate_from_precomputed_candidates(
    system_prompt: str,
    user_query: str,
    candidates: List[Dict[str, Any]],
    max_tokens: int = 6000,
    enforce_relevance: bool = True,
    relevance_threshold: float = MIN_RELEVANCE_SCORE
) -> Dict[str, Any]:
    """
    Generate the final response given precomputed candidate matches using REF ID format.
    
    This uses the APA 7 inline citation system:
    - Context shows sources with author names and years
    - LLM is instructed to use APA 7 format: Author et al. (Year)
    - References section lists all sources alphabetically
    
    IMPORTANT: This function now includes a relevance check to prevent
    forcing irrelevant citations when the database doesn't contain
    relevant data for the query topic.
    
    Args:
        system_prompt: Base system prompt
        user_query: User's query
        candidates: Precomputed candidate matches from hybrid search
        max_tokens: Maximum tokens for response
        enforce_relevance: If True, check relevance and refuse low-quality results
        relevance_threshold: Minimum score for relevance (default from MIN_RELEVANCE_SCORE)
    
    Returns:
        Dict with response, citations, and metadata
    """
    logger.info("=" * 60)
    logger.info("GENERATING FROM PRECOMPUTED CANDIDATES (REF ID FORMAT)")
    logger.info("=" * 60)

    # Deduplicate candidates by id while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        cid = str(c.get('id') or c.get('chunk_id', ''))
        if cid in seen or not cid:
            continue
        seen.add(cid)
        unique_candidates.append(c)
    
    logger.info(f"Processing {len(unique_candidates)} unique candidates")

    if not unique_candidates:
        return {
            'response': 'No relevant research found for this query.',
            'citations': [],
            'has_research': False,
            'num_sources': 0,
            'raw_chunks': [],
            'source': 'research'
        }
    
    # =========================================================
    # RELEVANCE CHECK - Prevent forcing irrelevant citations
    # =========================================================
    if enforce_relevance:
        relevance_check = check_query_relevance(
            query_text=user_query,
            candidates=unique_candidates,
            score_threshold=relevance_threshold
        )
        
        logger.info(f"Relevance check: is_relevant={relevance_check['is_relevant']}, "
                   f"avg_score={relevance_check['avg_score']:.3f}, "
                   f"max_score={relevance_check['max_score']:.3f}, "
                   f"relevant_count={relevance_check['relevant_count']}/{relevance_check['total_candidates']}")
        
        if not relevance_check["is_relevant"]:
            logger.warning(f"LOW RELEVANCE DETECTED - Retrieved documents do not match query topic")
            logger.warning(f"Detected domains in results: {relevance_check.get('detected_domains', [])}")
            logger.warning(f"Available domains: {relevance_check.get('available_domains', [])}")
            
            # Return a helpful response instead of forcing irrelevant citations
            return generate_no_relevant_data_response(user_query, relevance_check)

    # Build numbered context with REF ID format
    numbered_context, citations_data = build_numbered_ref_context(unique_candidates)

    # Enhanced system prompt with APA 7 citation instructions
    ref_id_instructions = """
## CITATION FORMAT REQUIREMENTS (APA 7 Inline Citations)

The research context below provides studies with author names and years.

**CRITICAL INSTRUCTIONS:**

1. **Data Density:** Extract specific statistics, percentages, ages, and sample sizes from the text. Do not generalize if specific numbers are available.

2. **MANDATORY: Use APA 7 inline citation format**
   - Single Author: (Smith, 2020)
   - Two authors: (Smith & Johnson, 2020)
   - Multiple authors: (Smith et al., 2020)
   - Multiple sources: (Smith, 2020; Jones et al., 2019)
   - **If no author name is available, do NOT include inline citation**
   - Do NOT use brackets with numbers like [1] or [2]

3. **Placement (Strict):** Citations must appear **ONLY at the very end of the sentence**, immediately before the period.
   - ✅ CORRECT: "The treatment group showed a 12% reduction in recidivism (Miller et al., 2015)."
   - ❌ WRONG (Mid-sentence): "The study (Miller et al., 2015) showed a reduction..."
   - ❌ WRONG (Narrative): "Miller et al. (2015) states that the group showed..."
   - **Correction Strategy:** If you want to say "Miller found X", rephrase it to "Research shows X (Miller, 2015)."

4. **Citation Matching:** Only cite sources that have author information in the research context.

5. **Do NOT generate a References section** - it will be automatically appended.

"""

    enhanced_system = f"""{system_prompt}

{ref_id_instructions}

## RESEARCH CONTEXT

{numbered_context}

---

Now provide your analysis using APA 7 inline citations (Author et al., Year).
"""

    logger.info("Generating AI response with REF ID context...")
    raw_response = safe_generate(
        system_msg=enhanced_system,
        user_msg=user_query,
        max_tokens=max_tokens
    )

    if not raw_response:
        logger.error("Generation failed for precomputed candidates")
        return {
            'response': 'Sorry, I could not generate a response. Please try again.',
            'citations': [],
            'has_research': False,
            'num_sources': 0,
            'raw_chunks': unique_candidates,
            'source': 'research'
        }

    # Build references section in APA 7 format (no numbered IDs)
    ref_lines = ["\n\n## References\n"]
    for cit in citations_data:
        full_citation = cit.get("full_citation", "")
        link = cit.get("link", "")
        
        if link:
            ref_lines.append(f"- {full_citation} Retrieved from {link}")
        else:
            ref_lines.append(f"- {full_citation}")
    
    ref_section = "\n".join(ref_lines)
    final_response = raw_response + ref_section

    logger.info(f"Response generated: {len(final_response)} chars, {len(citations_data)} citations")

    return {
        'response': final_response,
        'citations': citations_data,
        'has_research': len(unique_candidates) > 0,
        'num_sources': len(citations_data),
        'raw_chunks': unique_candidates,
        'source': 'research'
    }
