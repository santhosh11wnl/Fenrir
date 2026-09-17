"""Corpus ingestion.

Reads a project's ``ingest.yaml``, loads what it names, chunks it, embeds it,
and writes it to that project's collection. This is the per-project data step --
the thing that actually makes seven assistants different from each other.

Three source types:

``directory``  every matching file in a folder
``file``       one file
``url``        fetched over HTTP, cached on disk

PDFs, HTML, and Word files are extracted to text first (see
:mod:`~chatbot_core.retrieval.extractors`); a document that can't be extracted
honestly is skipped and reported rather than indexed as noise.

Re-ingest is idempotent: each document's existing chunks are deleted before its
new ones are written, so a document that shrank between runs doesn't leave
orphaned chunks that still match queries.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import ProjectConfig
from .retrieval import chunk_document, tokenizer_counter
from .retrieval.chunking import default_token_counter
from .retrieval.embeddings import LocalEmbedder, build_embedder
from .retrieval.extractors import SUPPORTED_SUFFIXES, extract, sniff_suffix
from .retrieval.store import Document, build_store, make_id

log = structlog.get_logger(__name__)

#: Where fetched URLs are cached, relative to the project directory. Cached so
#: re-ingesting doesn't re-download, and so an ingest is reproducible offline.
CACHE_DIRNAME = ".cache"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


class DirectorySource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["directory"] = "directory"
    path: str
    include: list[str] = Field(default_factory=lambda: ["**/*.md", "**/*.txt"])
    exclude: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FileSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["file"] = "file"
    path: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class URLSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["url"] = "url"
    url: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Re-download even if a cached copy exists.
    refresh: bool = False


#: The discriminator belongs on the union, not on the list field -- pydantic
#: rejects `list[...]` as a discriminated variant, and without it a URL source
#: would silently validate as a DirectorySource and fail far from the cause.
Source = Annotated[
    DirectorySource | FileSource | URLSource, Field(discriminator="type")
]


class IngestManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sources: list[Source] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> IngestManifest:
        path = Path(path)
        if not path.is_file():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoadedDocument:
    uri: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        line = f"{self.documents} documents -> {self.chunks} chunks"
        if self.skipped:
            line += f" ({len(self.skipped)} skipped)"
        return line


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


async def ingest_project(
    config: ProjectConfig, project_dir: str | Path, *, clean: bool = False
) -> IngestReport:
    """Ingest one project's corpus into its collection."""
    project_dir = Path(project_dir)
    if not config.retrieval.enabled:
        log.info("ingest_skipped_retrieval_disabled", project=config.project.id)
        return IngestReport()

    manifest = IngestManifest.load(project_dir / "ingest.yaml")
    if not manifest.sources:
        log.warning("ingest_no_sources", project=config.project.id)
        return IngestReport()

    embedder = build_embedder(config.retrieval.embedding)
    store = build_store(config.retrieval, config.collection, embedder)
    await store.ensure_ready()

    # Count with the embedding model's own tokenizer so chunks respect its real
    # input window rather than a character heuristic.
    count_tokens = (
        tokenizer_counter(embedder._model)  # noqa: SLF001 - same package
        if isinstance(embedder, LocalEmbedder)
        else default_token_counter
    )

    report = IngestReport()
    try:
        for doc in await _load_all(manifest.sources, project_dir, report):
            chunks = chunk_document(
                doc.text,
                config.retrieval.chunk,
                metadata=doc.metadata,
                count_tokens=count_tokens,
            )
            if not chunks:
                report.skipped.append(f"{doc.uri} (produced no chunks)")
                continue

            if not clean:
                await store.delete_by_uri(doc.uri)

            written = await store.upsert(
                [
                    Document(
                        id=make_id(doc.uri, chunk.index),
                        text=chunk.text,
                        title=doc.title,
                        uri=doc.uri,
                        metadata={**chunk.metadata, "chunk_index": chunk.index},
                    )
                    for chunk in chunks
                ]
            )
            report.documents += 1
            report.chunks += written
            log.info("ingested", uri=doc.uri, title=doc.title, chunks=written)
    finally:
        await store.aclose()

    log.info(
        "ingest_complete",
        project=config.project.id,
        documents=report.documents,
        chunks=report.chunks,
        skipped=len(report.skipped),
    )
    return report


async def _load_all(
    sources: Sequence[Source], project_dir: Path, report: IngestReport
) -> list[LoadedDocument]:
    docs: list[LoadedDocument] = []
    for source in sources:
        if isinstance(source, URLSource):
            doc = await _load_url(source, project_dir, report)
            if doc is not None:
                docs.append(doc)
        elif isinstance(source, DirectorySource):
            docs.extend(_load_directory(source, project_dir, report))
        else:
            doc = _load_file(
                (project_dir / source.path).resolve(),
                project_dir,
                source.metadata,
                report,
                title=source.title,
            )
            if doc is not None:
                docs.append(doc)
    return docs


def _load_directory(
    source: DirectorySource, project_dir: Path, report: IngestReport
) -> Iterator[LoadedDocument]:
    root = (project_dir / source.path).resolve()
    if not root.is_dir():
        report.skipped.append(f"{source.path} (not a directory)")
        log.warning("ingest_source_missing", path=str(root))
        return

    seen: set[Path] = set()
    for pattern in source.include:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, ex) for ex in source.exclude):
                continue
            doc = _load_file(path, project_dir, source.metadata, report)
            if doc is not None:
                yield doc


def _load_file(
    path: Path,
    project_dir: Path,
    metadata: dict[str, Any],
    report: IngestReport,
    *,
    title: str | None = None,
) -> LoadedDocument | None:
    if not path.is_file():
        report.skipped.append(f"{path} (not found)")
        return None
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        report.skipped.append(f"{path.name} (unsupported type {path.suffix or 'none'})")
        return None

    result = extract(path.read_bytes(), path.name)
    if result is None:
        report.skipped.append(f"{path.name} (no usable text extracted)")
        return None

    # URI is relative to the project directory so the index is portable between
    # a laptop and a container, where absolute paths differ.
    try:
        uri = path.relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        uri = path.as_posix()

    return LoadedDocument(
        uri=uri,
        title=title or result.title or _title_of(result.text, path.stem),
        text=result.text,
        metadata={**metadata, **(result.metadata or {}), "source": uri},
    )


async def _load_url(
    source: URLSource, project_dir: Path, report: IngestReport
) -> LoadedDocument | None:
    """Fetch a URL, caching the raw bytes so re-ingest doesn't re-download."""
    import httpx

    cache_dir = project_dir / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(source.url.encode()).hexdigest()[:16]

    existing = next(cache_dir.glob(f"{stem}.*"), None)
    if existing is not None and not source.refresh:
        data = existing.read_bytes()
        cached = existing
        log.info("url_cache_hit", url=source.url, bytes=len(data))
    else:
        try:
            async with httpx.AsyncClient(
                timeout=60.0, follow_redirects=True
            ) as client:
                response = await client.get(
                    source.url, headers={"User-Agent": "mcp-platform-ingest/0.1"}
                )
                response.raise_for_status()
                data = response.content
                content_type = response.headers.get("content-type", "")
        except httpx.HTTPError as exc:
            report.skipped.append(f"{source.url} ({exc})")
            log.warning("url_fetch_failed", url=source.url, error=str(exc))
            return None

        # Name the cache file by what the bytes actually are, not by the URL
        # path. `arxiv.org/pdf/1706.03762` has no extension, and guessing from
        # the path would hand a PDF to the HTML extractor -- which "succeeds",
        # producing mojibake that passes a length check and poisons the index.
        suffix = sniff_suffix(data, name=urlparse(source.url).path, content_type=content_type)
        cached = cache_dir / f"{stem}{suffix}"
        cached.write_bytes(data)
        log.info(
            "url_fetched", url=source.url, bytes=len(data), detected=suffix
        )

    result = extract(data, cached.name)
    if result is None:
        report.skipped.append(f"{source.url} (no usable text extracted)")
        return None

    return LoadedDocument(
        # The URL is the identity, not the cache path -- so a citation links
        # back to the original and re-ingest updates rather than duplicates.
        uri=source.url,
        title=source.title or result.title or _title_of(result.text, source.url),
        text=result.text,
        metadata={**source.metadata, **(result.metadata or {}), "source": source.url},
    )




def _title_of(text: str, fallback: str) -> str:
    """Prefer the document's own first heading; fall back to a readable name."""
    for line in text.splitlines()[:40]:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped.startswith("## ") and not stripped.startswith("## Page "):
            return stripped[3:].strip()
    name = Path(fallback).stem or fallback
    return name.replace("-", " ").replace("_", " ").strip().title() or fallback


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: ``python -m chatbot_core.ingest projects/<name>``"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ingest a project's corpus")
    parser.add_argument("project_dir", help="Path to the project directory")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Skip per-document deletion (only safe on a freshly dropped collection)",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    config = ProjectConfig.load(project_dir / "config.yaml")
    report = asyncio.run(ingest_project(config, project_dir, clean=args.clean))

    print(f"{config.project.id}: {report}")
    for item in report.skipped:
        print(f"  skipped: {item}", file=sys.stderr)


if __name__ == "__main__":
    main()


__all__ = [
    "DirectorySource",
    "FileSource",
    "IngestManifest",
    "IngestReport",
    "URLSource",
    "ingest_project",
]
