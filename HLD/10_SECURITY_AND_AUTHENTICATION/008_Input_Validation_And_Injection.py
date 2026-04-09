"""
INPUT VALIDATION AND INJECTION ATTACKS
=========================================

Problem Statement:
Injections (SQL, NoSQL, Command, LDAP, XSS) are consistently the #1
web security risk (OWASP Top 10). They occur when untrusted data is
sent to an interpreter without proper sanitization.

SQL Injection:
  Attacker inserts SQL into input field:
    Input: "'; DROP TABLE users; --"
    Query: SELECT * FROM users WHERE name = ''; DROP TABLE users; --'
  Prevention:
    Parameterized queries / prepared statements (PRIMARY defense).
    ORM with query builders (SQLAlchemy, Hibernate).
    Principle of least privilege on DB user.
    Input validation (secondary defense).

Parameterized Query:
  cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
  Driver handles escaping. SQL structure and data are separate.
  NEVER use string formatting/concatenation for SQL.

XSS (Cross-Site Scripting):
  Inject JavaScript into pages served to other users.
  Stored XSS: persisted in DB, served to all visitors.
  Reflected XSS: returned immediately in response.
  DOM XSS: client-side JS creates DOM from untrusted data.
  Prevention:
    Output encoding: HTML entity encoding for untrusted data.
    Content Security Policy (CSP) header.
    HttpOnly cookies (JS can't steal session).
    innerHTML → textContent; avoid eval().

Command Injection:
  os.system("ping " + user_input) with "google.com; rm -rf /"
  Prevention: Never pass user input to shell. Use subprocess with list args.

Path Traversal:
  File access: "../../etc/passwd"
  Prevention: Canonicalize path; verify starts with allowed base directory.

SSRF (Server-Side Request Forgery):
  Make server fetch internal resources: http://169.254.169.254/
  (AWS EC2 metadata endpoint)
  Prevention: Allowlist outbound URLs; block internal IP ranges.

NoSQL Injection:
  MongoDB: {"user": {"$gt": ""}} → matches all users.
  Prevention: Use typed schemas, validate field types, use ORM.

Input Validation Strategy:
  Validate at boundaries (API edge, not internal functions).
  Type checking: ensure int is int, not "1; DROP TABLE..."
  Length limits: reject inputs exceeding max length.
  Allowlist over blocklist: permit known good patterns.
  Fail closed: reject if uncertain.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import re
import os
import html
import urllib.parse
import hashlib
import sqlite3
import tempfile


# ─────────────────────────────────────────────
# SQL INJECTION DEMOS
# ─────────────────────────────────────────────

class VulnerableDB:
    """Demonstrates SQL injection via string concatenation (NEVER DO THIS)."""

    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        cur = self._conn.cursor()
        cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, admin INTEGER DEFAULT 0)")
        cur.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 1)")
        cur.execute("INSERT INTO users VALUES (2, 'Bob',   'bob@example.com',   0)")
        self._conn.commit()

    def get_user_UNSAFE(self, username: str) -> List:
        """VULNERABLE: string formatting with user input."""
        query = f"SELECT id, name, admin FROM users WHERE name = '{username}'"
        try:
            cur = self._conn.cursor()
            cur.execute(query)
            return cur.fetchall()
        except Exception as e:
            return [f"ERROR: {e}"]

    def get_user_SAFE(self, username: str) -> List:
        """SAFE: parameterized query."""
        cur = self._conn.cursor()
        cur.execute("SELECT id, name, admin FROM users WHERE name = ?", (username,))
        return cur.fetchall()


# ─────────────────────────────────────────────
# INPUT VALIDATOR
# ─────────────────────────────────────────────

@dataclass
class ValidationRule:
    field_name : str
    required   : bool = True
    min_length : int   = 0
    max_length : int   = 1000
    pattern    : str   = None    # regex pattern
    allowed_values: Optional[Set] = None
    strip      : bool  = True


@dataclass
class ValidationResult:
    valid   : bool
    errors  : List[str] = field(default_factory=list)
    cleaned : Dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """
    Allowlist-based input validation.
    Validates types, lengths, patterns.
    """

    def __init__(self, rules: List[ValidationRule]):
        self._rules = {r.field_name: r for r in rules}

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        errors  = []
        cleaned = {}

        for field_name, rule in self._rules.items():
            value = data.get(field_name)
            if value is None or value == "":
                if rule.required:
                    errors.append(f"{field_name}: required field missing")
                continue

            if isinstance(value, str) and rule.strip:
                value = value.strip()

            # Length check
            if isinstance(value, str):
                if len(value) < rule.min_length:
                    errors.append(f"{field_name}: too short (min {rule.min_length})")
                    continue
                if len(value) > rule.max_length:
                    errors.append(f"{field_name}: too long (max {rule.max_length})")
                    continue

            # Pattern check
            if rule.pattern and isinstance(value, str):
                if not re.fullmatch(rule.pattern, value):
                    errors.append(f"{field_name}: invalid format")
                    continue

            # Allowlist
            if rule.allowed_values and value not in rule.allowed_values:
                errors.append(f"{field_name}: value not allowed")
                continue

            cleaned[field_name] = value

        return ValidationResult(valid=len(errors) == 0, errors=errors,
                                 cleaned=cleaned)


# ─────────────────────────────────────────────
# XSS PREVENTION
# ─────────────────────────────────────────────

class XSSPrevention:
    """HTML output encoding to prevent XSS."""

    HTML_CHARS = {
        "&"  : "&amp;",
        "<"  : "&lt;",
        ">"  : "&gt;",
        '"'  : "&quot;",
        "'"  : "&#x27;",
        "/"  : "&#x2F;",
    }

    def encode_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return html.escape(text, quote=True)

    def encode_url(self, text: str) -> str:
        """URL-encode for use in URLs."""
        return urllib.parse.quote(text, safe="")

    def encode_js(self, text: str) -> str:
        """Escape for embedding in JavaScript strings."""
        result = ""
        for char in text:
            if char.isalnum() or char in " ,.!?":
                result += char
            else:
                result += f"\\u{ord(char):04x}"
        return result

    def safe_render(self, template: str, data: Dict[str, str]) -> str:
        """Render template with HTML-encoded values."""
        for key, value in data.items():
            safe_value = self.encode_html(str(value))
            template   = template.replace(f"{{{{{key}}}}}", safe_value)
        return template

    @staticmethod
    def csp_header() -> str:
        """Strong Content Security Policy header."""
        return (
            "default-src 'self'; "
            "script-src 'self'; "          # no inline scripts
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "     # clickjacking protection
            "base-uri 'self'; "
            "form-action 'self';"
        )


# ─────────────────────────────────────────────
# PATH TRAVERSAL PREVENTION
# ─────────────────────────────────────────────

class SafeFileAccess:
    """Prevents path traversal attacks."""

    def __init__(self, allowed_base: str):
        self._base = os.path.realpath(allowed_base)

    def safe_read(self, user_filename: str) -> Tuple[bool, str]:
        """Returns (safe, reason_or_content)."""
        # Normalize and resolve
        requested = os.path.realpath(os.path.join(self._base, user_filename))
        # Ensure path is within allowed base
        if not requested.startswith(self._base + os.sep) and requested != self._base:
            return False, f"path_traversal_blocked: {user_filename!r}"
        if not os.path.exists(requested):
            return False, "file_not_found"
        try:
            with open(requested, "r") as f:
                return True, f.read()
        except Exception as e:
            return False, f"read_error: {e}"


# ─────────────────────────────────────────────
# COMMAND INJECTION PREVENTION
# ─────────────────────────────────────────────

class SafeShellExec:
    """Prevents command injection via argument list (not shell string)."""

    ALLOWED_COMMANDS = {"ping", "nslookup", "dig"}

    def safe_ping(self, host: str) -> Tuple[bool, str]:
        """Safe: uses list of args, no shell expansion."""
        import subprocess
        import re
        # Allowlist: hostname pattern only
        if not re.fullmatch(r"[a-zA-Z0-9.\-]{1,253}", host):
            return False, f"invalid_hostname: {host!r}"
        # subprocess with list = no shell injection
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", host],
                capture_output=True, text=True, timeout=3
            )
            return True, result.stdout[:200]
        except Exception as e:
            return False, str(e)


# ─────────────────────────────────────────────
# DEMONSTRATION
# ─────────────────────────────────────────────

def demonstrate_injection_prevention():
    print("=" * 65)
    print("INPUT VALIDATION AND INJECTION ATTACKS")
    print("=" * 65)

    # ── SQL Injection ──────────────────────────────
    print("\n[1] SQL INJECTION — VULNERABLE vs SAFE")
    print("─" * 55)

    db = VulnerableDB()

    normal    = "Alice"
    injection = "' OR '1'='1"
    admin_byp = "' OR admin=1 --"

    print(f"  UNSAFE queries:")
    for payload in [normal, injection, admin_byp]:
        rows = db.get_user_UNSAFE(payload)
        print(f"    Input: {payload!r:30} → {rows}")

    print(f"\n  SAFE queries (parameterized):")
    for payload in [normal, injection, admin_byp]:
        rows = db.get_user_SAFE(payload)
        print(f"    Input: {payload!r:30} → {rows}")

    # ── Input Validation ──────────────────────────
    print("\n\n[2] INPUT VALIDATION — ALLOWLIST")
    print("─" * 55)

    validator = InputValidator([
        ValidationRule("username", required=True, min_length=3, max_length=30,
                        pattern=r"[a-zA-Z0-9_]+"),
        ValidationRule("email",    required=True, max_length=254,
                        pattern=r"[^@]+@[^@]+\.[^@]+"),
        ValidationRule("age",      required=False),
        ValidationRule("role",     required=True,
                        allowed_values={"user", "moderator"}),
    ])

    test_inputs = [
        {"username": "alice42", "email": "alice@example.com",  "role": "user"},
        {"username": "a",       "email": "not-an-email",       "role": "admin"},   # invalid
        {"username": "'; DROP TABLE users;--", "email": "x@y.z", "role": "user"},   # SQLi attempt
        {"username": "bob",     "email": "bob@x.com",         "role": "user", "age": "25"},
    ]
    for inp in test_inputs:
        result = validator.validate(inp)
        print(f"  Input: {str(inp)[:50]}")
        print(f"    valid={result.valid}  errors={result.errors}")

    # ── XSS Prevention ────────────────────────────
    print("\n\n[3] XSS PREVENTION — OUTPUT ENCODING")
    print("─" * 55)

    xss = XSSPrevention()
    payloads = [
        "<script>alert('xss')</script>",
        'javascript:alert("xss")',
        '<img src=x onerror="alert(1)">',
        "<a href='#' onclick='steal()'>Click me</a>",
    ]
    for p in payloads:
        encoded = xss.encode_html(p)
        print(f"  Original: {p}")
        print(f"  Encoded:  {encoded}\n")

    template = "<p>Welcome, {{name}}! Your score is {{score}}.</p>"
    rendered = xss.safe_render(template, {
        "name" : "<script>alert(1)</script>",
        "score": "100 <b>bonus</b>"
    })
    print(f"  Template rendering with XSS attempt:")
    print(f"  {rendered}")

    print(f"\n  CSP Header:")
    print(f"  Content-Security-Policy: {xss.csp_header()[:80]}...")

    # ── Path Traversal ────────────────────────────
    print("\n\n[4] PATH TRAVERSAL PREVENTION")
    print("─" * 55)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a safe file
        with open(os.path.join(tmpdir, "public.txt"), "w") as f:
            f.write("public content")

        sfa = SafeFileAccess(tmpdir)
        traversals = [
            "public.txt",
            "../../etc/passwd",
            "../secret.txt",
            "/etc/hosts",
            "subdir/../../../etc/shadow",
        ]
        for path in traversals:
            safe, result = sfa.safe_read(path)
            print(f"  Path: {path:<35} → {'ALLOWED' if safe else 'BLOCKED'}: {result[:40]}")

    # ── Security Headers Checklist ────────────────
    print("\n\n[5] SECURITY HEADERS CHECKLIST")
    print("─" * 55)

    headers = [
        ("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload"),
        ("Content-Security-Policy",   "default-src 'self'; ..."),
        ("X-Content-Type-Options",    "nosniff"),
        ("X-Frame-Options",           "DENY"),
        ("Referrer-Policy",           "strict-origin-when-cross-origin"),
        ("Permissions-Policy",        "geolocation=(), camera=(), microphone=()"),
        ("Cache-Control",             "no-store (for sensitive pages)"),
    ]
    for header, value in headers:
        print(f"  {header:<36} {value}")


if __name__ == "__main__":
    demonstrate_injection_prevention()
