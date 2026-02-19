"""
Document intelligence tool for Doris.

Extract structured information from PDFs and documents using langextract + LLM.
Supports local Ollama (qwen3:8b) with Gemini 2.5 Flash fallback.
"""

import os
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
import glob as glob_module

# Lazy imports to avoid loading heavy dependencies at startup
_pymupdf = None
_langextract = None

def _get_pymupdf():
    global _pymupdf
    if _pymupdf is None:
        import pymupdf
        _pymupdf = pymupdf
    return _pymupdf

def _get_langextract():
    global _langextract
    if _langextract is None:
        import langextract as lx
        _langextract = lx
    return _langextract


# Database path for extraction cache
CACHE_DB_PATH = Path(__file__).parent.parent / "data" / "document_cache.db"

# Directories to search for documents — configure for your system
SEARCH_DIRECTORIES = [
    Path.home() / "Documents",
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs",
]

# Maximum text length to send to LLM (avoid token limits)
MAX_TEXT_LENGTH = 15000

# Sensitive file patterns — never search, read, or return these.
# Prevents LLM-directed exfiltration of credentials, keys, and tokens.
import fnmatch as _fnmatch

SENSITIVE_FILE_PATTERNS = [
    ".env", ".env.*",                           # Environment files with secrets
    "*.pem", "*.key", "*.p8", "*.p12", "*.pfx", # Private keys / certificates
    "*.jks", "*.keystore",                       # Java keystores
    "token.json", "token.json.*",                # OAuth tokens (incl. encrypted)
    "credentials.json", "credentials*.json",     # Google / service credentials
    "id_rsa*", "id_ed25519*", "id_ecdsa*",       # SSH private keys
    "known_hosts", "authorized_keys",            # SSH infrastructure
    ".netrc", ".npmrc", ".pypirc",               # Package manager credentials
    ".htpasswd",                                 # HTTP basic auth
    "*.gpg", "*.pgp",                            # Encrypted key material
    "service_account*.json",                     # GCP service accounts
    "client_secret*.json",                       # OAuth client secrets
]


def _is_sensitive_file(path: Path) -> bool:
    """Check if a file matches any sensitive file pattern."""
    name = path.name
    return any(_fnmatch.fnmatch(name, pat) for pat in SENSITIVE_FILE_PATTERNS)


def _is_within_allowed_dirs(resolved_path: Path) -> bool:
    """Check if a resolved path is within the allowed search directories.

    Prevents symlink traversal attacks where a symlink inside ~/Documents
    points to ~/.ssh or the Doris project root, escaping the directory jail.
    """
    return any(
        resolved_path == d or resolved_path.is_relative_to(d)
        for d in SEARCH_DIRECTORIES
    )


def init_document_cache():
    """Initialize SQLite cache for document extractions."""
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_extractions (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            file_mtime REAL NOT NULL,
            schema_name TEXT NOT NULL,
            extraction_json TEXT NOT NULL,
            source_text TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(file_path, schema_name)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_file_path ON document_extractions(file_path)
    """)

    conn.commit()
    conn.close()


def _generate_cache_id(file_path: str, schema_name: str) -> str:
    """Generate a unique cache ID for a file + schema combination."""
    key = f"{file_path}:{schema_name}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def get_cached_extraction(file_path: str, schema_name: str) -> Optional[dict]:
    """
    Get cached extraction if file hasn't changed.

    Args:
        file_path: Path to the document
        schema_name: Name of the extraction schema used

    Returns:
        Cached extraction dict or None if cache miss
    """
    init_document_cache()

    try:
        current_mtime = os.path.getmtime(file_path)
    except OSError:
        return None

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()

    cursor.execute("""
        SELECT extraction_json, file_mtime
        FROM document_extractions
        WHERE file_path = ? AND schema_name = ?
    """, (file_path, schema_name))

    row = cursor.fetchone()
    conn.close()

    if row:
        extraction_json, cached_mtime = row
        # Check if file has been modified since cache
        if abs(current_mtime - cached_mtime) < 1.0:  # 1 second tolerance
            try:
                return json.loads(extraction_json)
            except json.JSONDecodeError:
                pass

    return None


def cache_extraction(file_path: str, schema_name: str, extraction: dict, source_text: str = None):
    """
    Cache an extraction result.

    Args:
        file_path: Path to the document
        schema_name: Name of the extraction schema used
        extraction: Extraction result dict
        source_text: Optional source text (truncated for storage)
    """
    init_document_cache()

    cache_id = _generate_cache_id(file_path, schema_name)

    try:
        file_mtime = os.path.getmtime(file_path)
    except OSError:
        return

    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cursor = conn.cursor()

    # Truncate source text for storage
    if source_text and len(source_text) > 5000:
        source_text = source_text[:5000] + "..."

    cursor.execute("""
        INSERT OR REPLACE INTO document_extractions
        (id, file_path, file_mtime, schema_name, extraction_json, source_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        cache_id,
        file_path,
        file_mtime,
        schema_name,
        json.dumps(extraction),
        source_text,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    Extract text content from a file.

    Supports:
    - PDF files (via pymupdf)
    - Text files (.txt, .md, .csv)

    Args:
        file_path: Path to the file

    Returns:
        Extracted text or None if extraction failed
    """
    path = Path(file_path)

    if not path.exists():
        return None

    # Resolve symlinks to prevent traversal attacks (e.g., ~/Documents/link -> ~/.ssh/)
    resolved = path.resolve()

    # Refuse to read files outside allowed directories (symlink escape)
    if not _is_within_allowed_dirs(resolved):
        import logging
        logging.getLogger(__name__).warning(
            "Blocked file read — resolved path %s is outside allowed directories", resolved
        )
        return None

    # Refuse to read sensitive files (credentials, keys, tokens)
    # Check both the original name and the resolved name
    if _is_sensitive_file(path) or _is_sensitive_file(resolved):
        return None

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            pymupdf = _get_pymupdf()
            doc = pymupdf.open(file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            text = "\n".join(text_parts)
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            return "\n".join(lines)
        except Exception as e:
            print(f"[documents] PDF extraction error: {e}")
            return None

    elif suffix in (".txt", ".md", ".csv", ".json"):
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[documents] Text file read error: {e}")
            return None

    else:
        # Attempt to read as text anyway
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None


def extract_from_document(
    text: str,
    schema: dict,
    query: Optional[str] = None,
    use_ollama: bool = True
) -> dict:
    """
    Extract structured information from document text using langextract.

    Args:
        text: Document text content
        schema: Extraction schema dict
        query: Optional specific question to answer
        use_ollama: If True, try Ollama first, then fall back to Gemini

    Returns:
        Extraction result dict
    """
    from schemas.document_schemas import build_extraction_prompt

    # Truncate text if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH] + "\n... (truncated)"

    prompt = build_extraction_prompt(schema, text, query)

    # Try Ollama first (local, free)
    if use_ollama:
        result = _extract_with_ollama(prompt)
        if result and _is_valid_extraction(result):
            return result

    # Fall back to Gemini
    result = _extract_with_gemini(prompt)
    if result:
        return result

    return {"error": "Extraction failed with all backends"}


def _extract_with_ollama(prompt: str) -> Optional[dict]:
    """Extract using local Ollama with qwen3:8b."""
    import httpx

    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:8b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 3000  # Needs to be high for thinking models
                }
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()

        response_text = data.get("response", "")

        # Extract JSON from response (handle markdown code blocks)
        json_str = _extract_json_from_text(response_text)
        if json_str:
            return json.loads(json_str)

    except Exception as e:
        print(f"[documents] Ollama extraction error: {e}")

    return None


def _extract_with_gemini(prompt: str) -> Optional[dict]:
    """Extract using Gemini 2.5 Flash API."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[documents] No GOOGLE_API_KEY set, skipping Gemini fallback")
        return None

    try:
        from google import genai
        from google.genai.types import GenerateContentConfig

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt + "\n\nRespond with ONLY valid JSON, no other text.",
            config=GenerateContentConfig(temperature=0.1, max_output_tokens=2000)
        )

        response_text = response.text
        json_str = _extract_json_from_text(response_text)
        if json_str:
            return json.loads(json_str)

    except Exception as e:
        print(f"[documents] Gemini extraction error: {e}")

    return None


def _extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()

    # Try to find JSON in code block
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()

    # Try to find raw JSON object
    if "{" in text:
        start = text.find("{")
        # Find matching closing brace
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]

    return None


def _is_valid_extraction(result: dict) -> bool:
    """Check if extraction result has meaningful content."""
    if not result or "error" in result:
        return False

    # Check if at least some fields have non-null values
    non_null_count = sum(1 for v in result.values() if v is not None and v != "" and v != [])
    return non_null_count >= 2


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}


def search_files(
    query: str,
    file_pattern: Optional[str] = None,
    schema_name: Optional[str] = None
) -> list[Path]:
    """
    Search for documents matching the query.

    Args:
        query: Natural language query (used to infer document type)
        file_pattern: Optional specific filename pattern
        schema_name: Optional schema name to get patterns from

    Returns:
        List of matching file paths, sorted by relevance
    """
    from schemas.document_schemas import get_schema_for_query, get_search_patterns_for_schema

    # Get schema to determine search patterns
    if schema_name:
        schema = {"name": schema_name}
        patterns = get_search_patterns_for_schema(schema_name)
    else:
        schema = get_schema_for_query(query)
        patterns = get_search_patterns_for_schema(schema["name"])

    # Extract keywords from query to search in filenames
    import re
    query_patterns = []
    for word in query.split():
        if len(word) < 3:
            continue
        # Strip punctuation but preserve original case
        original_word = re.sub(r'[^\w]', '', word)
        if not original_word:
            continue
        word_lower = original_word.lower()
        # Skip common words
        if word_lower in ("what", "when", "where", "which", "does", "have", "much", "many", "from", "about", "this", "that", "with", "your", "their", "paid"):
            continue
        # Add original, lowercase, title case, and uppercase patterns
        query_patterns.extend([
            f"*{original_word}*",  # Original case (e.g., CityPay)
            f"*{word_lower}*",     # lowercase
            f"*{word_lower.title()}*",  # Title case
            f"*{word_lower.upper()}*"   # UPPERCASE
        ])

    # Add user-specified pattern
    if file_pattern:
        patterns = [file_pattern, f"{file_pattern}.pdf", f"{file_pattern}.PDF"] + patterns

    # Add query-derived patterns
    patterns = query_patterns + patterns

    found_files = []
    seen_paths = set()

    for directory in SEARCH_DIRECTORIES:
        if not directory.exists():
            continue

        for pattern in patterns:
            # Search recursively
            for path in directory.rglob(pattern):
                # Only include supported file types
                if not path.is_file():
                    continue
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                # Resolve symlinks and verify the real path is still
                # within allowed directories (prevents symlink escape)
                resolved = path.resolve()
                resolved_key = str(resolved)
                if resolved_key in seen_paths:
                    continue
                if not _is_within_allowed_dirs(resolved):
                    continue

                # Block sensitive files — check both symlink name and target
                if _is_sensitive_file(path) or _is_sensitive_file(resolved):
                    continue

                # Skip SDK/library directories (check resolved path)
                resolved_str = str(resolved)
                skip_patterns = [
                    "/google-cloud-sdk/",
                    "/node_modules/",
                    "/.venv/",
                    "/site-packages/",
                    "/Library/Caches/",
                    "/.Trash/",
                ]
                if any(skip in resolved_str for skip in skip_patterns):
                    continue

                seen_paths.add(resolved_key)
                found_files.append(resolved)

    # Score files by keyword match relevance
    def score_file(path: Path) -> tuple:
        filename_lower = path.name.lower()
        # Count how many query words appear in the filename
        keyword_matches = sum(1 for word in query_words if word in filename_lower)
        # Also check original words (preserve case matches as bonus)
        for word in query.split():
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word in path.name:
                keyword_matches += 2  # Bonus for exact case match
        mtime = path.stat().st_mtime
        return (-keyword_matches, -mtime)  # Sort by matches desc, then mtime desc

    query_words = [re.sub(r'[^\w]', '', w.lower()) for w in query.split() if len(w) > 3]
    found_files.sort(key=score_file)

    # Limit results
    return found_files[:20]


def search_and_extract(
    query: str,
    file_pattern: Optional[str] = None,
    max_files: int = 3
) -> dict:
    """
    Main entry point: search for documents and extract relevant information.

    Args:
        query: Natural language question about documents
        file_pattern: Optional specific filename pattern
        max_files: Maximum number of files to extract from

    Returns:
        Dict with extractions and metadata
    """
    from schemas.document_schemas import get_schema_for_query

    # Determine schema based on query
    schema = get_schema_for_query(query)
    schema_name = schema["name"]

    # Search for matching files
    files = search_files(query, file_pattern, schema_name)

    if not files:
        return {
            "success": False,
            "error": "No matching documents found",
            "query": query,
            "schema": schema_name,
            "extractions": []
        }

    extractions = []

    for file_path in files[:max_files]:
        file_str = str(file_path)

        # Check cache first
        cached = get_cached_extraction(file_str, schema_name)
        if cached:
            extractions.append({
                "path": file_str,
                "filename": file_path.name,
                "extraction": cached,
                "cached": True
            })
            continue

        # Extract text from file
        text = extract_text_from_file(file_str)
        if not text:
            continue

        # Run extraction
        extraction = extract_from_document(text, schema, query)

        if _is_valid_extraction(extraction):
            # Cache the result
            cache_extraction(file_str, schema_name, extraction, text)

            extractions.append({
                "path": file_str,
                "filename": file_path.name,
                "extraction": extraction,
                "cached": False
            })

    return {
        "success": len(extractions) > 0,
        "query": query,
        "schema": schema_name,
        "files_searched": len(files),
        "extractions": extractions
    }


def format_extraction(result: dict, query: str) -> str:
    """
    Format extraction result as natural language.

    Args:
        result: Result from search_and_extract()
        query: Original user query

    Returns:
        Natural language response summarizing the extraction
    """
    if not result.get("success") or not result.get("extractions"):
        return "I couldn't find any documents matching that query."

    # Get the first (most relevant) extraction
    extraction = result["extractions"][0]
    data = extraction.get("extraction", {})
    filename = extraction.get("filename", "document")

    query_lower = query.lower()

    # Format based on what was asked
    response_parts = []

    # Insurance queries
    if "policy" in query_lower and "number" in query_lower:
        policy_num = data.get("policy_number") or data.get("member_id")
        if policy_num:
            response_parts.append(f"Your policy number is {policy_num}")

    elif "premium" in query_lower:
        premium = data.get("premium")
        if premium:
            response_parts.append(f"Your premium is {premium}")

    elif "deductible" in query_lower:
        deductible = data.get("deductible")
        if deductible:
            response_parts.append(f"Your deductible is {deductible}")

    # Vehicle registration queries
    elif "registration" in query_lower and "expir" in query_lower:
        exp_date = data.get("expiration_date")
        if exp_date:
            response_parts.append(f"Your registration expires {exp_date}")

    elif "plate" in query_lower or "license" in query_lower:
        plate = data.get("plate_number")
        if plate:
            response_parts.append(f"Your plate number is {plate}")

    elif "vin" in query_lower:
        vin = data.get("vin")
        if vin:
            response_parts.append(f"Your VIN is {vin}")

    # Receipt queries
    elif "total" in query_lower or "paid" in query_lower or "pay" in query_lower or "cost" in query_lower or "spent" in query_lower:
        total = data.get("total_amount")
        merchant = data.get("merchant_name", "")
        date = data.get("transaction_date", "")
        if total:
            # Format amount nicely
            if isinstance(total, (int, float)):
                total = f"${total:.2f}"
            if merchant and date:
                response_parts.append(f"You paid {total} to {merchant} on {date}")
            elif merchant:
                response_parts.append(f"You paid {total} to {merchant}")
            else:
                response_parts.append(f"The total was {total}")

    # Generic fallback - summarize key fields
    if not response_parts:
        # Find the most relevant non-null fields
        key_fields = []
        for key, value in data.items():
            if value and value not in [None, "", [], "null"]:
                # Format key nicely
                key_formatted = key.replace("_", " ")
                key_fields.append(f"{key_formatted}: {value}")

        if key_fields:
            response_parts.append(f"From {filename}, I found: " + ", ".join(key_fields[:5]))
        else:
            return "I found a document but couldn't extract the specific information you asked for."

    return ". ".join(response_parts)


# Initialize cache on module load
init_document_cache()
