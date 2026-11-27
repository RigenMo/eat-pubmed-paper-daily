"""Fail-closed matching to dated, publisher-supplied two-year JIF evidence."""
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import requests

NATURE_METRICS = "https://www.nature.com/nature-portfolio/about-journals/journal-metrics"
NATURE_SOURCE = NATURE_METRICS + "#citation-impact"


def journal_key(title):
    from unicodedata import normalize
    return " ".join(re.findall(r"\w+", normalize("NFKC", title or "").casefold().replace("&", " and ")))


class _NatureTables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_heading = False
        self.in_metrics = False
        self.heading = []
        self.year = None
        self.row = None
        self.cell = None
        self.jif_column = None
        self.results = {}
        self.conflicts = set()

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self.in_metrics = False
            self.in_heading = dict(attrs).get("id") == "citation-impact"
            self.heading = []
        elif self.in_metrics and tag == "table":
            self.jif_column = None
        elif self.in_metrics and tag == "tr":
            self.row = []
        elif self.row is not None and tag in {"td", "th"}:
            self.cell = []

    def handle_data(self, data):
        if self.in_heading:
            self.heading.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "h2" and self.in_heading:
            m = re.fullmatch(r"\s*(20\d\d)\s+Journal Metrics\s*", "".join(self.heading))
            if m:
                self.year = int(m[1])
                self.in_metrics = True
            self.in_heading = False
        elif self.cell is not None and tag in {"td", "th"}:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            row, self.row = self.row, None
            if "Journal Impact Factor" in row:
                self.jif_column = row.index("Journal Impact Factor")
            elif (self.jif_column is not None and self.jif_column > 0 and len(row) > self.jif_column
                  and re.fullmatch(r"\d+(?:\.\d+)?", row[self.jif_column])):
                name = journal_key(row[0])
                record = {"jif": float(row[self.jif_column]), "jif_year": self.year, "source": NATURE_SOURCE}
                if name in self.results and self.results[name] != record:
                    self.conflicts.add(name)
                self.results[name] = record


def parse_nature_metrics(html):
    parser = _NatureTables()
    parser.feed(html)
    return {k: v for k, v in parser.results.items() if k not in parser.conflicts}


def load_nature_metrics():
    try:
        response = requests.get(NATURE_METRICS, timeout=8, allow_redirects=True)
        parsed = urlsplit(response.url)
        if response.ok and parsed.scheme == "https" and parsed.hostname == "www.nature.com":
            return parse_nature_metrics(response.text)
    except (requests.RequestException, ValueError):
        pass
    return {}
