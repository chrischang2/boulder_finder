"""
Boulder name detection and matching in text captions.

Finds boulder names by locating V-grades (e.g. V3, V10) and extracting
the preceding words back to the last punctuation boundary, capped at
5 words.  This captures *any* boulder name, not just known ones.

Known boulders (from data/train/ folders) are still recognised so that
folder names stay consistent across scrapes.
"""

import re
from pathlib import Path

# Regex to strip emoji and other non-text Unicode symbols
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"              # zero-width joiner
    "\U000020E3"              # combining enclosing keycap
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000200B"              # zero-width space
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    """Remove emoji characters from *text*."""
    return _EMOJI_RE.sub("", text).strip()

# ── Auto-discover known boulders from training folders ───────────────
_TRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "train"

EXTRA_BOULDERS: list[str] = [
    # Add boulder names here that aren't in data/train/ yet, e.g.:
    # "quantum", "kryptonite", "sperm whale",
]


def _discover_boulders() -> dict[str, str]:
    """
    Return {lowercase_display_name: folder_name} for every boulder.

    Sources:
      1. Sub-folders of data/train/
      2. EXTRA_BOULDERS list above
    """
    boulders: dict[str, str] = {}

    if _TRAIN_DIR.is_dir():
        for p in sorted(_TRAIN_DIR.iterdir()):
            if p.is_dir():
                display = p.name.replace("_", " ")
                boulders[display.lower()] = p.name

    for name in EXTRA_BOULDERS:
        folder = name.strip().lower().replace(" ", "_")
        boulders[name.strip().lower()] = folder

    return boulders


# {display_name_lower: folder_name}  e.g. {"the joker": "the_joker"}
KNOWN_BOULDERS: dict[str, str] = _discover_boulders()


# ── Grade regex ──────────────────────────────────────────────────────
# Matches V-grades: V0, V9, V10, V14, V3+  (case-insensitive)
_VGRADE_RE = re.compile(r"\bV(\d{1,2}\+?)\b", re.IGNORECASE)

# Characters that act as a "sentence boundary" when walking backwards
_PUNCT = set(".,;:!?\"'()[]{}#@\n\r\t|/\\")

# Regex for characters to strip from folder names
_FOLDER_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")

# Words that are clearly NOT part of a boulder name (stop words).
# NOTE: 'the' is intentionally excluded — it appears in real names
# like 'The Joker'
_STOP_WORDS = {
    "a", "an", "on", "at", "in", "to", "for", "and", "or",
    "but", "is", "was", "my", "our", "its", "with", "of", "from",
    "i", "we", "he", "she", "it", "they", "me", "us", "him", "her",
    "just", "also", "been", "had", "have", "has", "got", "did", "do",
    "finally", "tried", "sent", "topped", "climbed", "flashed",
    "session", "today", "yesterday", "then", "so", "some",
    "great", "amazing", "sick", "nice", "good", "cool",
    "working", "moves", "move",
}

_MAX_NAME_WORDS = 5


# ── Public API ───────────────────────────────────────────────────────

class BoulderMatch:
    """A single boulder name and grade found in text."""

    __slots__ = ("name", "grade", "folder_name")

    def __init__(self, name: str, grade: str, folder_name: str):
        self.name = name            # Display name, e.g. "The Joker"
        self.grade = grade          # e.g. "V9"
        self.folder_name = folder_name  # e.g. "the_joker"

    def __repr__(self) -> str:
        return (f"BoulderMatch({self.name!r}, {self.grade}, "
                f"folder={self.folder_name!r})")

    @property
    def label(self) -> str:
        """Human-friendly label like 'The Joker V9'."""
        return f"{self.name.title()} {self.grade}"


def _extract_name_before(text: str, start: int) -> str | None:
    """
    Walk backwards from *start* in *text* to extract the boulder name.

    Stops at punctuation or beginning of string.  Takes up to
    _MAX_NAME_WORDS words, stripping leading stop-words.
    """
    # Grab the chunk of text before the V-grade
    chunk = text[:start].rstrip()
    if not chunk:
        return None

    # Walk backwards to the nearest punctuation boundary
    boundary = len(chunk)
    for i in range(len(chunk) - 1, -1, -1):
        if chunk[i] in _PUNCT:
            boundary = i + 1
            break
    else:
        boundary = 0  # no punctuation found → start of string

    segment = chunk[boundary:].strip()
    if not segment:
        return None

    # Remove any embedded V-grades AND the connector word that follows
    # e.g. "Cyclops v4 and Submarine Boulder" → "Cyclops  Submarine Boulder"
    # Only strip connector if it immediately follows a removed V-grade.
    segment = re.sub(
        r'\bV\d{1,2}\+?\s*(?:and|or)\b',
        ' ',
        segment,
        flags=re.IGNORECASE,
    )
    # Remove any remaining standalone V-grades
    segment = re.sub(r'\bV\d{1,2}\+?\b', ' ', segment, flags=re.IGNORECASE)
    segment = re.sub(r'\s+', ' ', segment).strip()
    if not segment:
        return None

    # Strip emoji characters from extracted name
    segment = _strip_emojis(segment)
    if not segment:
        return None

    # Split into words, strip trailing connectors left after V-grade removal
    words = segment.split()
    while words and words[-1].lower() in {"and", "or", "then", "but"}:
        words.pop()
    # Take the last N words
    words = words[-_MAX_NAME_WORDS:]

    # Strip leading stop-words (keep trailing ones — they can be part
    # of the name, e.g. "lock and load pull")
    while words and words[0].lower() in _STOP_WORDS:
        words.pop(0)

    # Strip trailing tokens that are pure punctuation (e.g. lone "-")
    while words and not any(c.isalnum() for c in words[-1]):
        words.pop()

    if not words:
        return None

    return " ".join(words)


def find_boulders(text: str) -> list[BoulderMatch]:
    """
    Search *text* for boulder names by locating V-grades and extracting
    the preceding words as the boulder name.

    Returns a list of ``BoulderMatch`` objects — (name, grade).
    Duplicate names are de-duplicated (first occurrence wins).
    """
    if not text:
        return []

    seen: set[str] = set()
    matches: list[BoulderMatch] = []

    for m in _VGRADE_RE.finditer(text):
        grade = f"V{m.group(1).upper()}" if m.group(1)[-1] == '+' \
            else f"V{m.group(1)}"

        raw_name = _extract_name_before(text, m.start())
        if raw_name is None:
            continue

        key = raw_name.lower()
        if key in seen:
            continue
        seen.add(key)

        # Use canonical folder name if this is a known boulder,
        # otherwise build a sanitised folder name
        folder = KNOWN_BOULDERS.get(key)
        if folder is None:
            folder = _FOLDER_SANITIZE_RE.sub("_", key)
            folder = folder.strip("_")
            # collapse runs of underscores
            folder = re.sub(r"_+", "_", folder)

        matches.append(BoulderMatch(
            name=raw_name,
            grade=grade,
            folder_name=folder,
        ))

    return matches


def has_boulder_match(text: str) -> bool:
    """Return True if *text* contains at least one boulder + grade."""
    return bool(find_boulders(text))


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Finally sent The Joker V9 today! #castlehillbouldering",
        "Working the moves on Ode to Joy V7",
        "Beautiful day at Castle Hill, tried Cyclops v4 and Submarine Boulder V8",
        "No boulders mentioned here, just a hike",
        "had a great session on top heavy V3.",
        "finally topped the joker V9",
        "tuppi master V6 is so classic",
        "Top Heavy V3",
        "Lock and Load Pull V5 was amazing, then did Kryptonite V10",
        "Great day! sent quantum field theory boulder V12 in one go",
    ]
    print("Known boulders:", list(KNOWN_BOULDERS.keys()))
    print()
    for s in samples:
        hits = find_boulders(s)
        print(f"  {s!r}")
        for h in hits:
            print(f"    → name={h.name!r}  grade={h.grade}  folder={h.folder_name}")
        if not hits:
            print(f"    → (no match)")
        print()
