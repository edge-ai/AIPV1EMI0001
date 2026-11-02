import re


PRIVATE_REFERENCE_PATTERNS = (
    ("private network address",
     r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b(?!-\d)"),
    ("station serial number", r"\b[A-Z]{3}\d[A-Z]\d{6}\b"),
    ("personal home directory", r"/(?:home|Users)/[^/\s\"']+"),
    ("session workspace", r"\b(?:agent|session)[-_]workspace\b"),
    ("SSH login", r"\b(?:ssh|scp|rsync)\s+[a-z_][\w.-]*@[a-z0-9][\w.-]*"),
)


def assert_no_private_references(test, text, name):
    for description, pattern in PRIVATE_REFERENCE_PATTERNS:
        if re.search(pattern, text):
            test.fail(f"{name} contains a {description}")
