"""MkDocs hooks.

Docstrings across the package annotate cross-references with Sphinx roles
(``:meth:`fit```, ``:class:`sklearn.cluster.DBSCAN```). MkDocs has no notion of
those roles, so without this the role prefix renders as literal text in front of
every cross-reference. Rewriting them here keeps the docstrings usable under
``help()`` and Sphinx while rendering cleanly on the site.
"""

from __future__ import annotations

import re
from typing import Any

# Matches a role prefix immediately followed by the code span that markdown has
# already produced from the backticks, so a bare ``:meth:`` in prose is left be.
_SPHINX_ROLE = re.compile(
    r":(?:class|meth|func|mod|attr|obj|data|exc|ref|py:\w+):(?=<code>)"
)


def on_page_content(html: str, **_: Any) -> str:
    """Strip Sphinx role prefixes from rendered cross-references."""

    return _SPHINX_ROLE.sub("", html)
