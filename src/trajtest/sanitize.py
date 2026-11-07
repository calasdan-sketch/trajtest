"""Make untrusted text safe to print in CI logs and terminals.

Agent output (notes, tool args, error messages, URLs) is untrusted. Printed
raw it can inject ANSI escapes, forge fake report lines with newlines, or
visually reorder text with bidi controls. It can also carry credentials the
agent was handed. `safe()` neutralises the first three; `redact()` masks the
last.
"""

import re
import unicodedata

REDACTED = "«redacted»"

# Keys whose values are masked in transcripts (case-insensitive substring).
_SECRET_KEY = (
    r"authorization|api[-_ ]?key|apikey|x-api-key|access[-_ ]?token|"
    r"refresh[-_ ]?token|auth[-_ ]?token|token|secret|passw(?:or)?d|passwd|"
    r"cookie|credential|private[-_ ]?key"
)
# 'key': 'value'  /  "key": "value"   (dict reprs, JSON)
_DICT_PAIR = re.compile(
    r"""(?P<k>(['"])[^'"]{0,40}?(?:"""
    + _SECRET_KEY
    + r""")[^'"]{0,40}?\2\s*:\s*)(?P<q>['"]).*?(?:(?<!\\)(?P=q)|$)""",
    re.IGNORECASE,
)
# key='value'  (kwargs reprs)
_KWARG_PAIR = re.compile(
    r"""(?P<k>\b[\w-]{0,40}?(?:"""
    + _SECRET_KEY
    + r""")[\w-]{0,40}?\s*=\s*)(?P<q>['"]).*?(?:(?<!\\)(?P=q)|$)""",
    re.IGNORECASE,
)
# Well-known credential shapes, masked wherever they appear (positional
# args, free text). Not exhaustive: see README limits.
_SHAPES = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{10,}|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}"
    r"|github_pat_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{35}|npm_[A-Za-z0-9]{30,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})"
)
# ?api_key=VALUE / &token=VALUE in URLs and form bodies (unquoted values)
_QUERY_PAIR = re.compile(
    r"(?P<k>[?&;][\w.-]{0,40}?(?:" + _SECRET_KEY + r"|sig(?:nature)?|auth)[\w.-]{0,40}?=)"
    r"[^&\s'\"#]+",
    re.IGNORECASE,
)
# scheme://user:password@host
_USERINFO = re.compile(r"(?P<k>\b[a-z][a-z0-9+.-]{1,20}://[^\s/:@'\"]{1,100}:)[^\s/@'\"]{1,200}(?=@)", re.IGNORECASE)
_PEM = re.compile(r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----.*?(?:-----END [A-Z ]{0,30}PRIVATE KEY-----|$)", re.DOTALL)
_BEARER = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{6,}", re.IGNORECASE)


def redact(text: str) -> str:
    """Mask values under secret-looking keys and Bearer/Basic credentials."""
    text = _PEM.sub(REDACTED, text)
    text = _QUERY_PAIR.sub(lambda m: f"{m['k']}{REDACTED}", text)
    text = _USERINFO.sub(lambda m: f"{m['k']}{REDACTED}", text)
    text = _DICT_PAIR.sub(lambda m: f"{m['k']}{m['q']}{REDACTED}{m['q']}", text)
    text = _KWARG_PAIR.sub(lambda m: f"{m['k']}{m['q']}{REDACTED}{m['q']}", text)
    text = _BEARER.sub(lambda m: f"{m[1]} {REDACTED}", text)
    return _SHAPES.sub(REDACTED, text)


def safe(text) -> str:
    """Escape control, format (bidi / zero-width), and line-separator chars.

    Printable text is untouched. A newline becomes the two characters ``\\n``
    so untrusted text can never forge an extra report line.
    """
    out = []
    for ch in str(text):
        cat = unicodedata.category(ch)
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cat in ("Cc", "Cf", "Zl", "Zp", "Cs", "Co", "Cn"):
            cp = ord(ch)
            out.append(f"\\x{cp:02x}" if cp < 0x100 else f"\\u{cp:04x}" if cp < 0x10000 else f"\\U{cp:08x}")
        else:
            out.append(ch)
    return "".join(out)


def clip(text, limit: int = 2000) -> str:
    """Bound untrusted text BEFORE the regex passes (a 20 MB value used to
    take 15 s to print). Keeps head and tail so a secret cut in half at the
    boundary is still mostly masked by shape rules on each side."""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[: limit // 2]}…(+{len(text) - limit} chars)…{text[-limit // 2 :]}"


def clean(text, limit: int = 20000) -> str:
    """clip, redact() then safe(): the form used for everything we print."""
    return safe(redact(clip(text, limit)))
