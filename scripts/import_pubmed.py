"""Optional abstract-only PubMed importer (no copyrighted full-text downloads).

NCBI recommends including email/tool identity and respecting the public E-utilities rate limit.
Requires network access and terms-of-use compliance for the retrieved abstracts.
"""
from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET

import httpx

from eeg_agent_rag.core.config import Settings
from eeg_agent_rag.core.container import build_services

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def request_xml(client: httpx.Client, endpoint: str, params: dict) -> ET.Element:
    response = client.get(f"{EUTILS}/{endpoint}", params=params)
    response.raise_for_status()
    return ET.fromstring(response.content)


def flatten(node: ET.Element | None) -> str:
    return " ".join(" ".join(node.itertext()).split()) if node is not None else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="PubMed search query (e.g. EEG AND depression)")
    parser.add_argument("--email", default=os.getenv("NCBI_EMAIL", ""))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--api-key", default=os.getenv("NCBI_API_KEY", ""))
    args = parser.parse_args()
    if "@" not in args.email:
        parser.error("Provide a valid --email for NCBI E-utilities")
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be in 1..100")
    common = {"tool": "eeg-agent-rag", "email": args.email}
    if args.api_key:
        common["api_key"] = args.api_key
    with httpx.Client(timeout=45) as client:
        root = request_xml(client, "esearch.fcgi", {
            **common, "db": "pubmed", "term": args.query,
            "retmax": args.limit, "retmode": "xml", "sort": "relevance",
        })
        identifiers = [x.text for x in root.findall("./IdList/Id") if x.text]
        if not identifiers:
            print("No PubMed results")
            return
        time.sleep(0.4 if not args.api_key else 0.15)
        xml = request_xml(client, "efetch.fcgi", {
            **common, "db": "pubmed", "id": ",".join(identifiers), "retmode": "xml",
        })
    services = build_services(Settings())
    for entry in xml.findall("./PubmedArticle"):
        article = entry.find("./MedlineCitation/Article")
        pmid = flatten(entry.find("./MedlineCitation/PMID"))
        if article is None or not pmid:
            continue
        title = flatten(article.find("./ArticleTitle"))
        abstract = "\n".join(
            (f"{x.get('Label')}: " if x.get("Label") else "") + flatten(x)
            for x in article.findall("./Abstract/AbstractText") if flatten(x)
        )
        if not abstract:
            continue
        doi = ""
        for item in entry.findall("./PubmedData/ArticleIdList/ArticleId"):
            if item.get("IdType") == "doi":
                doi = flatten(item)
        year_text = flatten(article.find("./Journal/JournalIssue/PubDate/Year"))
        try:
            year = int(year_text)
        except ValueError:
            year = 0
        body = (f"PMID: {pmid}\nTITLE: {title}\nDOI: {doi}\n\nABSTRACT:\n{abstract}").encode("utf-8")
        info = services.indexing.index_bytes(
            filename=f"PMID-{pmid}.txt", content=body,
            source="pubmed-abstract", doi=doi, year=year,
            license="source-terms-apply", title=title,
        )
        print(info.document_id, "PMID", pmid, info.status)


if __name__ == "__main__":
    main()
