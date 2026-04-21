"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import logging

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    tiktoken = None
    _tokenizer = None

from pyzotero import zotero

from .chroma_client import ChromaClient, create_chroma_client
from .chunking import chunk_fulltext
from .client import get_zotero_client
from .utils import format_creators, is_local_mode
from .local_db import LocalZoteroReader, get_local_zotero_reader

logger = logging.getLogger(__name__)


from zotero_mcp.utils import suppress_stdout


def _truncate_to_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to fit within embedding model token limit.

    Uses tiktoken for accurate token counting when available,
    falls back to conservative character-based estimation.
    """
    if _tokenizer is not None:
        tokens = _tokenizer.encode(text, disallowed_special=())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = _tokenizer.decode(tokens)
    else:
        # Fallback: conservative char limit (~1.5 chars/token for non-Latin scripts)
        max_chars = max_tokens * 2
        if len(text) > max_chars:
            text = text[:max_chars]
    return text


class CrossEncoderReranker:
    """Optional cross-encoder re-ranker for semantic search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[int]:
        """Re-rank documents by relevance to query.

        Returns indices of top_k documents in descending relevance order.
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices[:top_k]


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""

    def __init__(self,
                 chroma_client: ChromaClient | None = None,
                 config_path: str | None = None,
                 db_path: str | None = None):
        """
        Initialize semantic search.

        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
            db_path: Optional path to Zotero database (overrides config file)
        """
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        self.db_path = db_path  # CLI override for Zotero database path

        # Load update configuration
        self.update_config = self._load_update_config()

        # Reranker (lazy-initialized on first search)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_config = self._load_reranker_config()

        # Scope for the active update_database run (collection_key, collection_name).
        # Set by update_database(); read by _create_metadata to stamp records.
        self._current_scope: tuple[str | None, str | None] = (None, None)

        # Chunked-schema marker: check + seed on empty, warn once on legacy.
        self._check_schema_marker()

    _CHUNK_SCHEMA_VERSION = 1

    def _check_schema_marker(self) -> None:
        """Emit one WARN on a legacy non-empty collection; seed marker on empty.

        The marker (``chunked: True``) lives on the Chroma collection's
        metadata and tells us whether this index was built with the chunked
        pipeline. Without it on a populated collection, retrieval is
        degraded (one vector per paper), so we point the user at
        ``update-db --force-rebuild``.
        """
        try:
            col = self.chroma_client.collection
            meta = getattr(col, "metadata", None) or {}
            if meta.get("chunked") is True:
                return  # Already chunked — nothing to do.

            count = 0
            try:
                count = col.count()
            except Exception:
                count = 0

            if count == 0:
                # Fresh / empty collection — seed the marker silently.
                try:
                    col.modify(metadata={
                        "chunked": True,
                        "chunk_version": self._CHUNK_SCHEMA_VERSION,
                    })
                except Exception as e:
                    logger.debug(f"Could not seed chunked marker on empty collection: {e}")
            else:
                logger.warning(
                    "Legacy (unchunked) semantic search index detected: "
                    "records pre-date passage-level retrieval. "
                    "Run `zotero-mcp update-db --force-rebuild` to re-index "
                    "with chunked embeddings for better review-writing results."
                )
        except Exception as e:
            # Never crash init over the marker check.
            logger.debug(f"Schema marker check failed: {e}")

    def _set_chunked_marker(self) -> None:
        """Set the chunked marker on the current collection metadata."""
        try:
            col = self.chroma_client.collection
            current = dict(getattr(col, "metadata", None) or {})
            current.update({
                "chunked": True,
                "chunk_version": self._CHUNK_SCHEMA_VERSION,
            })
            col.modify(metadata=current)
        except Exception as e:
            logger.debug(f"Could not set chunked marker: {e}")

    def _load_chunking_config(self) -> dict[str, Any]:
        """Override-only config for fulltext chunking.

        Returns a dict with optional keys ``target_tokens`` / ``overlap_tokens``.
        Actual defaults are derived from the embedding model's token cap at
        runtime, so this only carries user overrides.
        """
        config: dict[str, Any] = {}
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(
                        file_config.get("semantic_search", {}).get("chunking", {})
                    )
            except Exception as e:
                logger.warning(f"Error loading chunking config: {e}")
        return config

    def _compute_chunk_sizes(self) -> tuple[int, int]:
        """Compute (target_tokens, overlap_tokens) for this embedding model.

        Derived from `chroma_client.embedding_max_tokens`, then overridden
        by any values in semantic_search.chunking in config.json.

        The 16-token safety margin keeps the post-truncate payload
        comfortably under the embedding model's hard cap even when the
        tokenizer's char-vs-token estimate drifts.
        """
        max_tokens = int(self.chroma_client.embedding_max_tokens or 2000)
        target = min(800, max(max_tokens - 16, 64))
        overlap = max(50, target // 8)

        overrides = self._load_chunking_config()
        if "target_tokens" in overrides:
            target = int(overrides["target_tokens"])
        if "overlap_tokens" in overrides:
            overlap = int(overrides["overlap_tokens"])

        # Ensure invariant: overlap strictly less than target.
        if overlap >= target:
            overlap = max(target // 4, 1)
        return target, overlap

    def _load_reranker_config(self) -> dict[str, Any]:
        """Load reranker configuration from file or use defaults.

        Reranker is ON by default: chunked retrieval surfaces many near-dup
        passages per paper, and a cross-encoder's query-passage pair scoring
        is dramatically better than raw cosine for ranking the top candidates
        for review-writing. First use downloads ~80MB model; users can opt
        out via ``semantic_search.reranker.enabled = false`` in config.json.
        """
        config: dict[str, Any] = {
            "enabled": True,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "candidate_multiplier": 3,
        }
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("reranker", {}))
            except Exception as e:
                logger.warning(f"Error loading reranker config: {e}")
        return config

    def _get_reranker(self) -> CrossEncoderReranker | None:
        """Get the reranker instance, lazily initializing if enabled."""
        if not self._reranker_config.get("enabled", False):
            return None
        if self._reranker is None:
            model = self._reranker_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker = CrossEncoderReranker(model_name=model)
        return self._reranker

    def _load_update_config(self) -> dict[str, Any]:
        """Load update configuration from file or use defaults."""
        config = {
            "auto_update": False,
            "update_frequency": "manual",
            "last_update": None,
            "update_days": 7
        }

        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("update_config", {}))
            except Exception as e:
                logger.warning(f"Error loading update config: {e}")

        return config

    def _save_update_config(self) -> None:
        """Save update configuration to file."""
        if not self.config_path:
            return

        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        full_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    full_config = json.load(f)
            except Exception:
                pass

        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        full_config["semantic_search"]["update_config"] = self.update_config

        try:
            with open(self.config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving update config: {e}")

    def _create_document_text(self, item: dict[str, Any]) -> str:
        """
        Create searchable text from a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Combined text for embedding
        """
        data = item.get("data", {})

        # Extract key fields for semantic search
        title = data.get("title", "")
        abstract = data.get("abstractNote", "")

        # Format creators as text
        creators = data.get("creators", [])
        creators_text = format_creators(creators)

        # Additional searchable content
        extra_fields = []

        # Publication details
        if publication := data.get("publicationTitle"):
            extra_fields.append(publication)

        # Tags
        if tags := data.get("tags"):
            tag_text = " ".join([tag.get("tag", "") for tag in tags])
            extra_fields.append(tag_text)

        # Note content (if available)
        if note := data.get("note"):
            # Clean HTML from notes
            import re
            note_text = re.sub(r'<[^>]+>', '', note)
            extra_fields.append(note_text)

        # Combine all text fields
        text_parts = [title, creators_text, abstract] + extra_fields
        return " ".join(filter(None, text_parts))

    def _create_metadata(self, item: dict[str, Any]) -> dict[str, Any]:
        """
        Create metadata for a Zotero item.

        Args:
            item: Zotero item dictionary

        Returns:
            Metadata dictionary for ChromaDB
        """
        data = item.get("data", {})

        metadata = {
            "item_key": item.get("key", ""),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
        }
        # If fulltext was extracted (or attempted), mark it so incremental
        # updates don't keep re-trying items that failed extraction
        if data.get("fulltext"):
            metadata["has_fulltext"] = True
            if data.get("fulltextSource"):
                metadata["fulltext_source"] = data.get("fulltextSource")
        elif data.get("fulltext_attempted"):
            # Extraction was attempted but failed (timeout, empty, etc.)
            # Mark so we don't retry on every incremental update
            metadata["has_fulltext"] = "failed"

        # Add tags as a single string
        if tags := data.get("tags"):
            metadata["tags"] = " ".join([tag.get("tag", "") for tag in tags])
        else:
            metadata["tags"] = ""

        # Add citation key if available
        extra = data.get("extra", "")
        citation_key = ""
        for line in extra.split("\n"):
            if line.lower().startswith(("citation key:", "citationkey:")):
                citation_key = line.split(":", 1)[1].strip()
                break
        metadata["citation_key"] = citation_key

        # Stamp active subcollection scope so searches can filter by it.
        scope_key, scope_name = getattr(self, "_current_scope", (None, None))
        if scope_key:
            metadata["collection_key"] = scope_key
            if scope_name:
                metadata["collection_name"] = scope_name

        return metadata

    def should_update_database(self) -> bool:
        """Check if the database should be updated based on configuration."""
        if not self.update_config.get("auto_update", False):
            return False

        frequency = self.update_config.get("update_frequency", "manual")

        if frequency == "manual":
            return False
        elif frequency == "startup":
            return True
        elif frequency == "daily":
            last_update = self.update_config.get("last_update")
            if not last_update:
                return True

            last_update_date = datetime.fromisoformat(last_update)
            return datetime.now() - last_update_date >= timedelta(days=1)
        elif frequency.startswith("every_"):
            try:
                days = int(frequency.split("_")[1])
                last_update = self.update_config.get("last_update")
                if not last_update:
                    return True

                last_update_date = datetime.fromisoformat(last_update)
                return datetime.now() - last_update_date >= timedelta(days=days)
            except (ValueError, IndexError):
                return False

        return False

    def _get_items_from_source(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
        collection_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get items from either local database or API.

        When extract_fulltext=True, requires local mode (ZOTERO_LOCAL=true);
        raises RuntimeError if local mode is not enabled.
        Otherwise uses API (faster, metadata-only).

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists
            collection_key: Restrict to items directly in this Zotero collection.

        Returns:
            List of items in API-compatible format
        """
        if extract_fulltext:
            if not is_local_mode():
                raise RuntimeError(
                    "Fulltext extraction requires local mode but ZOTERO_LOCAL is not enabled. "
                    "Set ZOTERO_LOCAL=true or run 'zotero-mcp setup' to enable local mode."
                )
            return self._get_items_from_local_db(
                limit,
                extract_fulltext=extract_fulltext,
                chroma_client=chroma_client,
                force_rebuild=force_rebuild,
                collection_key=collection_key,
            )
        else:
            return self._get_items_from_api(limit, collection_key=collection_key)

    def _get_items_from_local_db(
        self,
        limit: int | None = None,
        extract_fulltext: bool = False,
        chroma_client: ChromaClient | None = None,
        force_rebuild: bool = False,
        collection_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get items from local Zotero database.

        Args:
            limit: Optional limit on number of items
            extract_fulltext: Whether to extract fulltext content
            chroma_client: ChromaDB client to check for existing documents (None to skip checks)
            force_rebuild: Whether to force extraction even if item exists
            collection_key: Restrict to items directly in this Zotero collection.

        Returns:
            List of items in API-compatible format
        """
        logger.info("Fetching items from local Zotero database...")

        try:
            # Load per-run config, including extraction limits and db path if provided
            pdf_max_pages = None
            pdf_timeout = 30
            zotero_db_path = self.db_path  # CLI override takes precedence
            # If semantic_search config file exists, prefer its setting
            try:
                if self.config_path and os.path.exists(self.config_path):
                    with open(self.config_path) as _f:
                        _cfg = json.load(_f)
                        semantic_cfg = _cfg.get('semantic_search', {})
                        extraction_cfg = semantic_cfg.get('extraction', {})
                        pdf_max_pages = extraction_cfg.get('pdf_max_pages')
                        pdf_timeout = extraction_cfg.get('pdf_timeout', 30)
                        # Use config db_path only if no CLI override
                        if not zotero_db_path:
                            zotero_db_path = semantic_cfg.get('zotero_db_path')
            except Exception:
                pass

            with suppress_stdout(), LocalZoteroReader(db_path=zotero_db_path, pdf_max_pages=pdf_max_pages, pdf_timeout=pdf_timeout) as reader:
                # Phase 1: fetch metadata only (fast)
                sys.stderr.write("Scanning local Zotero database for items...\n")
                local_items = reader.get_items_with_text(
                    limit=limit,
                    include_fulltext=False,
                    collection_key=collection_key,
                )
                candidate_count = len(local_items)
                sys.stderr.write(f"Found {candidate_count} candidate items.\n")

                # Optional deduplication: if preprint and journalArticle share a DOI/title, keep journalArticle
                # Build index by (normalized DOI or normalized title)
                def norm(s: str | None) -> str | None:
                    if not s:
                        return None
                    return "".join(s.lower().split())

                key_to_best = {}
                for it in local_items:
                    doi_key = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                    title_key = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None

                    def consider(k):
                        if not k:
                            return
                        cur = key_to_best.get(k)
                        # Prefer journalArticle over preprint; otherwise keep first
                        if cur is None:
                            key_to_best[k] = it
                        else:
                            prefer_types = {"journalArticle": 2, "preprint": 1}
                            cur_score = prefer_types.get(getattr(cur, "item_type", ""), 0)
                            new_score = prefer_types.get(getattr(it, "item_type", ""), 0)
                            if new_score > cur_score:
                                key_to_best[k] = it

                    consider(doi_key)
                    consider(title_key)

                # If a preprint loses against a journal article for same DOI/title, drop it
                filtered_items = []
                for it in local_items:
                    # If there is a journalArticle alternative for same DOI or title, and this is preprint, drop
                    if getattr(it, "item_type", None) == "preprint":
                        k_doi = ("doi", norm(getattr(it, "doi", None))) if getattr(it, "doi", None) else None
                        k_title = ("title", norm(getattr(it, "title", None))) if getattr(it, "title", None) else None
                        drop = False
                        for k in (k_doi, k_title):
                            if not k:
                                continue
                            best = key_to_best.get(k)
                            if best is not None and best is not it and getattr(best, "item_type", None) == "journalArticle":
                                drop = True
                                break
                        if drop:
                            continue
                    filtered_items.append(it)

                local_items = filtered_items
                total_to_extract = len(local_items)
                if total_to_extract != candidate_count:
                    try:
                        sys.stderr.write(f"After filtering/dedup: {total_to_extract} items to process. Extracting content...\n")
                    except Exception:
                        pass
                else:
                    try:
                        sys.stderr.write("Extracting content...\n")
                    except Exception:
                        pass

                # Phase 2: selectively extract fulltext only when requested
                if extract_fulltext:
                    extracted = 0
                    skipped_existing = 0
                    updated_existing = 0
                    items_to_process = []

                    consecutive_timeouts = 0
                    MAX_CONSECUTIVE_TIMEOUTS = 5
                    _extraction_stopped = False  # Set True when circuit breaker trips

                    total_local = len(local_items)
                    _skipped_pdfs = []  # Collect timeout/error names for summary
                    _skipped_failed = []  # Items skipped because extraction previously failed

                    # Show startup note
                    try:
                        sys.stderr.write(
                            "\n  Note: Most papers take 1-3 seconds. Some larger or complex PDFs\n"
                            "  may take up to 30 seconds. Password-protected or corrupted files\n"
                            "  will be skipped automatically. The system moves on to the next\n"
                            "  paper if a file can't be processed in time.\n\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass

                    # Temporarily suppress local_db logger to prevent timeout warnings
                    # from disrupting the progress line — we collect them ourselves
                    _local_db_logger = logging.getLogger("zotero_mcp.local_db")
                    _prev_level = _local_db_logger.level
                    _local_db_logger.setLevel(logging.CRITICAL)

                    for item_idx, it in enumerate(local_items, 1):
                        # Build display string: Author (Year) — Title
                        title = getattr(it, "title", "") or ""
                        creators = getattr(it, "creators", "") or ""
                        date = getattr(it, "date_added", "") or ""
                        first_author = ""
                        if creators:
                            first_author = creators.split(";")[0].split(",")[0].strip()
                            if first_author:
                                first_author += " et al." if ";" in creators else ""
                        year = ""
                        if date and len(date) >= 4:
                            year = date[:4]
                        citation = ""
                        if first_author and year:
                            citation = f"{first_author} ({year}) — "
                        elif first_author:
                            citation = f"{first_author} — "
                        display = f"{citation}{title}"
                        if len(display) > 60:
                            display = display[:57] + "..."

                        # Single-line progress with \r overwrite
                        # MUST fit within terminal width to prevent wrapping
                        try:
                            try:
                                term_width = os.get_terminal_size().columns
                            except (OSError, ValueError):
                                term_width = 80
                            # Build the line and truncate to terminal width - 1
                            # (- 1 to prevent the cursor from wrapping to next line)
                            max_len = term_width - 1
                            status_parts = []
                            if skipped_existing > 0:
                                status_parts.append(f"{skipped_existing} up to date")
                            if extracted > 0:
                                status_parts.append(f"{extracted} extracted")
                            status = f" ({', '.join(status_parts)})" if status_parts else ""
                            prefix = f"  Processing {item_idx}/{total_local}{status} — "
                            # Truncate display to fit remaining space
                            remaining = max_len - len(prefix) - 3  # -3 for "..."
                            if remaining > 0 and display and len(display) > remaining:
                                display = display[:remaining] + "..."
                            line = f"{prefix}{display or 'working...'}"
                            if len(line) > max_len:
                                line = line[:max_len]
                            sys.stderr.write(f"\r{line}{' ' * max(0, max_len - len(line))}")
                            sys.stderr.flush()
                        except Exception:
                            pass

                        should_extract = True

                        # CHECK IF ITEM ALREADY EXISTS (unless force_rebuild or no client)
                        if chroma_client and not force_rebuild:
                            existing_metadata = chroma_client.get_document_metadata(it.key)
                            if existing_metadata:
                                chroma_has_fulltext = existing_metadata.get("has_fulltext", False)
                                local_has_fulltext = len(reader.get_fulltext_meta_for_item(it.item_id)) > 0

                                # Skip if extraction previously failed AND the item hasn't been
                                # modified since (handles case where user replaces a bad PDF)
                                if chroma_has_fulltext == "failed":
                                    chroma_date = existing_metadata.get("date_modified", "")
                                    item_date = getattr(it, "date_modified", "") or ""
                                    if chroma_date == item_date:
                                        # Same modification date — don't retry failed extraction
                                        should_extract = False
                                        skipped_existing += 1
                                        _skipped_failed.append(display or f"item {it.key}")
                                    else:
                                        # Item was modified since last failure — retry
                                        updated_existing += 1
                                elif not chroma_has_fulltext and local_has_fulltext:
                                    # Document exists but lacks fulltext - we need to update it
                                    updated_existing += 1
                                else:
                                    should_extract = False
                                    skipped_existing += 1

                        if should_extract:
                            # Extract fulltext if item doesn't have it yet
                            # (skip if circuit breaker has tripped)
                            if not getattr(it, "fulltext", None) and not _extraction_stopped:
                                text = reader.extract_fulltext_for_item(it.item_id)
                                # Circuit breaker: stop PDF extraction after consecutive timeouts
                                if isinstance(text, tuple) and len(text) == 2 and text[1] == "timeout":
                                    _skipped_pdfs.append(display or f"item {it.key}")
                                    consecutive_timeouts += 1
                                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                                        logger.warning(
                                            f"Stopping PDF extraction after {MAX_CONSECUTIVE_TIMEOUTS} "
                                            f"consecutive timeouts — remaining items will use metadata only"
                                        )
                                        try:
                                            sys.stderr.write(
                                                f"\n  Warning: PDF extraction stopped after {MAX_CONSECUTIVE_TIMEOUTS} "
                                                f"consecutive timeouts. Remaining items will be indexed with "
                                                f"metadata only (titles, abstracts, authors).\n\n"
                                            )
                                        except Exception:
                                            pass
                                        _extraction_stopped = True
                                    # Don't skip the item — still add it with metadata only
                                    it._fulltext_attempted = True  # Mark so metadata knows extraction was tried
                                else:
                                    # Reset counter on successful extraction
                                    if text:
                                        consecutive_timeouts = 0
                                    if text:
                                        # Support new (text, source) return format
                                        if isinstance(text, tuple) and len(text) == 2:
                                            it.fulltext, it.fulltext_source = text[0], text[1]
                                        else:
                                            it.fulltext = text
                                    else:
                                        # Extraction returned empty — mark as attempted
                                        it._fulltext_attempted = True
                            extracted += 1
                            items_to_process.append(it)

                            # (progress shown inline above via \r)

                    # Restore local_db logger
                    _local_db_logger.setLevel(_prev_level)

                    # Clear progress line and show extraction summary
                    try:
                        sys.stderr.write(f"\r{' ' * 120}\r")  # Clear progress line
                        parts = [f"  Extraction complete: {extracted} items to index"]
                        if skipped_existing > 0:
                            parts.append(f"{skipped_existing} already up to date")
                        sys.stderr.write(", ".join(parts) + "\n")
                        if updated_existing > 0:
                            sys.stderr.write(f"  ({updated_existing} items updated with new fulltext)\n")
                        if _skipped_pdfs:
                            sys.stderr.write(f"  Skipped {len(_skipped_pdfs)} PDF(s) (timed out):\n")
                            for name in _skipped_pdfs:
                                sys.stderr.write(f"    - {name}\n")
                        if _skipped_failed:
                            sys.stderr.write(f"  {len(_skipped_failed)} item(s) skipped (PDF extraction previously failed):\n")
                            for name in _skipped_failed[:5]:  # Show first 5
                                sys.stderr.write(f"    - {name}\n")
                            if len(_skipped_failed) > 5:
                                sys.stderr.write(f"    ... and {len(_skipped_failed) - 5} more\n")
                            sys.stderr.write(f"  (To retry these, run with --force-rebuild)\n")
                    except Exception:
                        pass

                    # Replace local_items with filtered list
                    local_items = items_to_process
                else:
                    # Skip fulltext extraction for faster processing
                    for it in local_items:
                        it.fulltext = None
                        it.fulltext_source = None

                # Convert to API-compatible format
                api_items = []
                for item in local_items:
                    # Create API-compatible item structure
                    api_item = {
                        "key": item.key,
                        "version": 0,  # Local items don't have versions
                        "data": {
                            "key": item.key,
                            "itemType": getattr(item, 'item_type', None) or "journalArticle",
                            "title": item.title or "",
                            "abstractNote": item.abstract or "",
                            "extra": item.extra or "",
                            # Include fulltext only when extracted
                            "fulltext": getattr(item, 'fulltext', None) or "" if extract_fulltext else "",
                            "fulltextSource": getattr(item, 'fulltext_source', None) or "" if extract_fulltext else "",
                            # Flag if extraction was attempted but failed (timeout, empty)
                            "fulltext_attempted": getattr(item, '_fulltext_attempted', False),
                            "dateAdded": item.date_added,
                            "dateModified": item.date_modified,
                            "creators": self._parse_creators_string(item.creators) if item.creators else []
                        }
                    }

                    # Add notes if available
                    if item.notes:
                        api_item["data"]["notes"] = item.notes

                    api_items.append(api_item)

                logger.info(f"Retrieved {len(api_items)} items from local database")
                return api_items

        except Exception as e:
            logger.error(f"Error reading from local database: {e}")
            logger.info("Falling back to API...")
            return self._get_items_from_api(limit)

    def _parse_creators_string(self, creators_str: str) -> list[dict[str, str]]:
        """
        Parse creators string from local DB into API format.

        Args:
            creators_str: String like "Smith, John; Doe, Jane"

        Returns:
            List of creator objects
        """
        if not creators_str:
            return []

        creators = []
        for creator in creators_str.split(';'):
            creator = creator.strip()
            if not creator:
                continue

            if ',' in creator:
                last, first = creator.split(',', 1)
                creators.append({
                    "creatorType": "author",
                    "firstName": first.strip(),
                    "lastName": last.strip()
                })
            else:
                creators.append({
                    "creatorType": "author",
                    "name": creator
                })

        return creators

    def _get_items_from_api(
        self,
        limit: int | None = None,
        collection_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get items from Zotero API (original implementation).

        Args:
            limit: Optional limit on number of items
            collection_key: Restrict to items directly in this Zotero collection
                (uses pyzotero ``collection_items`` instead of ``items``).

        Returns:
            List of items from API
        """
        if collection_key:
            logger.info(f"Fetching items from Zotero API (collection {collection_key})...")
        else:
            logger.info("Fetching items from Zotero API...")

        # Fetch items in batches to handle large libraries
        batch_size = 100
        start = 0
        all_items = []

        while True:
            batch_params = {"start": start, "limit": batch_size}
            if limit and len(all_items) >= limit:
                break

            try:
                if collection_key:
                    items = self.zotero_client.collection_items(collection_key, **batch_params)
                else:
                    items = self.zotero_client.items(**batch_params)
            except Exception as e:
                if "Connection refused" in str(e):
                    error_msg = (
                        "Cannot connect to Zotero local API. Please ensure:\n"
                        "1. Zotero is running\n"
                        "2. Local API is enabled in Zotero Preferences > Advanced > Enable HTTP server\n"
                        "3. The local API port (default 23119) is not blocked"
                    )
                    raise Exception(error_msg) from e
                else:
                    raise Exception(f"Zotero API connection error: {e}") from e
            if not items:
                break

            # Filter out attachments and notes by default
            filtered_items = [
                item for item in items
                if item.get("data", {}).get("itemType") not in ["attachment", "note"]
            ]

            all_items.extend(filtered_items)
            start += batch_size

            if len(items) < batch_size:
                break

        if limit:
            all_items = all_items[:limit]

        logger.info(f"Retrieved {len(all_items)} items from API")
        return all_items

    def update_database(self,
                       force_full_rebuild: bool = False,
                       limit: int | None = None,
                       extract_fulltext: bool = False,
                       collection_key: str | None = None,
                       collection_name: str | None = None) -> dict[str, Any]:
        """
        Update the semantic search database with Zotero items.

        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            extract_fulltext: Whether to extract fulltext content from local database
            collection_key: Restrict to items directly in this Zotero collection.
                When set, metadata for newly-indexed items is stamped with
                ``collection_key`` (and ``collection_name`` when provided).
            collection_name: Human-readable collection name for metadata
                stamping and progress messages.

        Returns:
            Update statistics
        """
        logger.info("Starting database update...")
        start_time = datetime.now()

        # Stash scope so _create_metadata (called from _process_item_batch)
        # stamps each record. Clear on function exit regardless of outcome.
        self._current_scope = (collection_key, collection_name)

        stats = {
            "total_items": 0,
            "processed_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "recovered_items": 0,
            "skipped_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None
        }

        try:
            # Reset collection if force rebuild.
            # Scoped rebuild purges only the target collection's records —
            # wiping the whole DB would destroy other subcollections the user
            # already indexed.
            if force_full_rebuild:
                if collection_key:
                    logger.info(
                        f"Force rebuilding scoped collection {collection_key}..."
                    )
                    self.chroma_client.delete_documents_where(
                        {"collection_key": collection_key}
                    )
                else:
                    logger.info("Force rebuilding database...")
                    self.chroma_client.reset_collection()
                # After a rebuild the records are chunked — stamp the marker
                # so the legacy warning stops firing.
                self._set_chunked_marker()

            # Get all items from either local DB or API
            all_items = self._get_items_from_source(
                limit=limit,
                extract_fulltext=extract_fulltext,
                chroma_client=self.chroma_client if not force_full_rebuild else None,
                force_rebuild=force_full_rebuild,
                collection_key=collection_key,
            )

            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")
            # User-friendly progress reporting
            total = stats['total_items'] = len(all_items)
            try:
                sys.stderr.write(f"\nIndexing {total} items...\n\n")
                sys.stderr.flush()
            except Exception:
                pass

            # Process items in batches. Sized against OpenAI's 300k-tokens
            # per-request cap: real papers average ~10k tokens of chunks each
            # (≈12 chunks × 800 tokens, measured on an ai-icu-review run),
            # and outliers hit 25k+. 16 papers × 10k ≈ 160k typical and
            # 16 × 25k = 400k worst case — close to the cap but the failing-
            # batch retry path recovers individual docs sequentially if we
            # overshoot. Gemini's per-batch=100 cap is handled inside
            # GeminiEmbeddingFunction. Going bigger than this consistently
            # tripped max_tokens_per_request on text-embedding-3-large.
            batch_size = 16
            seen_items = 0
            _failed_docs = []  # Collect failures for end-of-run retry
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i:i + batch_size]

                # Show per-item progress within this batch
                for item in batch:
                    seen_items += 1
                    title = item.get("data", {}).get("title", "")
                    if title and len(title) > 60:
                        title = title[:57] + "..."
                    pct = int(seen_items / total * 100) if total else 0
                    try:
                        sys.stderr.write(f"\r  [{pct:3d}%] {seen_items}/{total} — {title or 'processing...'}")
                        sys.stderr.flush()
                    except Exception:
                        pass

                batch_stats = self._process_item_batch(batch, force_full_rebuild, _failed_docs)

                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]

                logger.info(f"Processed {seen_items}/{total} items (added: {stats['added_items']}, skipped: {stats['skipped_items']})")

            # Retry any documents that failed during the main run
            if _failed_docs:
                try:
                    sys.stderr.write(f"\r{' ' * 120}\r")
                    sys.stderr.write(f"\n  Retrying {len(_failed_docs)} failed items...\n")
                except Exception:
                    pass

                import time as _retry_time
                _retry_time.sleep(1)  # Brief pause before retry

                retry_ok = 0
                retry_fail = 0
                for doc, meta, doc_id in _failed_docs:
                    try:
                        self.chroma_client.upsert_documents([doc], [meta], [doc_id])
                        retry_ok += 1
                        stats["errors"] -= 1  # Remove from error count
                        # Don't classify as added vs updated — when the
                        # original batch failed, the add/update lookup never
                        # ran, so we don't know which category it belongs in.
                        # Track recovered items in their own bucket.
                        stats["recovered_items"] += 1
                    except Exception as e2:
                        retry_fail += 1
                        logger.error(f"Retry failed for {doc_id}: {e2}")

                try:
                    sys.stderr.write(f"  Retry: {retry_ok} recovered, {retry_fail} still failed\n")
                except Exception:
                    pass

            # Clear the progress line and show summary
            try:
                sys.stderr.write(f"\r{' ' * 120}\r")  # Clear line
                summary = (
                    f"  Done: {stats['processed_items']} indexed, "
                    f"{stats['skipped_items']} skipped, "
                    f"{stats['errors']} errors"
                )
                if stats["recovered_items"]:
                    summary += f", {stats['recovered_items']} recovered"
                sys.stderr.write(summary + "\n")
            except Exception:
                pass

            # Update last update time
            self.update_config["last_update"] = datetime.now().isoformat()
            self._save_update_config()

            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            stats["end_time"] = end_time.isoformat()

            logger.info(f"Database update completed in {stats['duration']}")
            return stats

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            stats["error"] = str(e)
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats

    def _process_item_batch(
        self,
        items: list[dict[str, Any]],
        force_rebuild: bool = False,
        _failed_docs: list | None = None,
    ) -> dict[str, int]:
        """Process a batch of items into chunked Chroma documents.

        Each paper emits:
          - id=``<key>#-1`` summary doc: title + abstract + creators + tags
          - id=``<key>#<i>`` fulltext chunks (i=0..N-1) when fulltext is present

        Stale records for the same item_key (including legacy pre-chunking
        ids = bare key) are purged before upsert so re-runs leave no orphans.
        This is skipped when force_rebuild=True because the collection has
        already been reset at that layer.

        _failed_docs: optional list (passed by reference from update_database)
        that collects (doc_text, metadata, doc_id) tuples for batches that fail
        mid-run. Without this, a NameError would crash the whole reindex and
        transient ChromaDB errors become fatal.

        Stats semantics: ``added``/``updated`` count PAPERS; ``errors`` counts
        the chunk-level docs deferred to retry when a batch upsert fails.
        """
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}

        target_tokens, overlap_tokens = self._compute_chunk_sizes()

        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []
        # Papers queued for this batch, with their pre-upsert had_prior flag.
        # Applied to added/updated stats only after upsert succeeds so batch
        # failures don't double-count alongside recovered_items in the retry
        # loop at update_database.
        pending_papers: list[tuple[str, bool]] = []

        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue

                # Build per-paper chunk triples (id, text, metadata).
                summary_text = self._create_document_text(item).strip()
                fulltext = (item.get("data", {}).get("fulltext") or "").strip()
                base_metadata = self._create_metadata(item)

                paper_docs: list[tuple[str, str, dict[str, Any]]] = []

                if summary_text:
                    summary_meta = dict(base_metadata)
                    summary_meta["chunk_idx"] = -1
                    paper_docs.append((f"{item_key}#-1", summary_text, summary_meta))

                if fulltext:
                    chunks = chunk_fulltext(
                        fulltext,
                        target_tokens=target_tokens,
                        overlap_tokens=overlap_tokens,
                        tokenizer=_tokenizer,
                    )
                    for i, chunk_text in enumerate(chunks):
                        cmeta = dict(base_metadata)
                        cmeta["chunk_idx"] = i
                        paper_docs.append((f"{item_key}#{i}", chunk_text, cmeta))

                if not paper_docs:
                    stats["skipped"] += 1
                    continue

                # Stamp chunk_total on every doc now that we know N.
                total_chunks = len(paper_docs)
                for _, _, meta in paper_docs:
                    meta["chunk_total"] = total_chunks

                # Detect prior state BEFORE the purge so added-vs-updated stats
                # reflect the paper's pre-run presence, not post-purge state.
                had_prior = False
                if not force_rebuild:
                    try:
                        existing = self.chroma_client.collection.get(
                            where={"item_key": item_key}, limit=1
                        )
                        had_prior = bool(existing.get("ids"))
                    except Exception:
                        # Best-effort check; fall back to added-count on error.
                        had_prior = False
                    # Purge stale chunks for this paper (and any legacy bare-id
                    # record with the same item_key metadata).
                    self.chroma_client.delete_documents_where({"item_key": item_key})

                # Safety-truncate each chunk per the embedding cap — chunk_fulltext
                # already targets ~max, but this defends against drift in the
                # char-based fallback path and double-encoding artifacts.
                for doc_id, chunk_text, meta in paper_docs:
                    documents.append(self.chroma_client.truncate_text(chunk_text))
                    metadatas.append(meta)
                    ids.append(doc_id)

                pending_papers.append((item_key, had_prior))
                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1

        # Upsert all chunks for this batch in one shot.
        if documents:
            try:
                self.chroma_client.upsert_documents(documents, metadatas, ids)
            except Exception as e:
                # Batch failed — collect failures for end-of-run retry.
                # ChromaDB's ONNX tokenizer can fail intermittently in bursts;
                # retrying immediately usually fails too. Collecting failures
                # and retrying after all batches are done is more effective.
                logger.warning(f"Batch upsert failed ({e}), saving for retry")
                if _failed_docs is not None:
                    for j in range(len(documents)):
                        _failed_docs.append((documents[j], metadatas[j], ids[j]))
                    # Count chunk-level docs as errors for stats accuracy;
                    # the retry loop decrements per successful recovery.
                    stats["errors"] += len(documents)
                    # Do NOT bump added/updated here — the retry loop at
                    # update_database increments recovered_items instead.
                    return stats
                else:
                    # No retry list — re-raise instead of silently swallowing.
                    raise

            # Upsert succeeded — apply paper-level added/updated stats now.
            for _, had_prior in pending_papers:
                if had_prior:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1

        return stats

    def search(self,
               query: str,
               limit: int = 10,
               filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Perform semantic search over the Zotero library.

        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            Search results with Zotero item details
        """
        try:
            # Over-fetch candidates: chunked indexing means several chunks
            # from the same paper may hit — we need headroom for grouping to
            # still yield `limit` unique papers. Rerank adds another multiplier.
            reranker = self._get_reranker()
            group_multiplier = 4  # absorbs typical per-paper chunk count
            fetch_limit = limit * group_multiplier
            if reranker:
                multiplier = self._reranker_config.get("candidate_multiplier", 3)
                fetch_limit = max(fetch_limit, limit * multiplier)

            # Perform semantic search
            results = self.chroma_client.search(
                query_texts=[query],
                n_results=fetch_limit,
                where=filters
            )

            # Re-rank results with cross-encoder if enabled. Rerank runs on
            # raw chunks before grouping — empirically simpler and lets the
            # grouping step just pick the best chunk per paper post-rerank.
            if reranker and results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                # Keep more rerank candidates than `limit` so grouping has room.
                rerank_top_k = min(len(documents), fetch_limit)
                ranked_indices = reranker.rerank(query, documents, top_k=rerank_top_k)
                for key in ["ids", "distances", "documents", "metadatas"]:
                    if results.get(key) and results[key][0]:
                        results[key][0] = [results[key][0][i] for i in ranked_indices]

            # Enrich results with full Zotero item data (grouped by paper).
            enriched_results = self._enrich_search_results(results, query)

            # Cap to `limit` unique papers after grouping.
            enriched_results = enriched_results[:limit]

            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": enriched_results,
                "total_found": len(enriched_results)
            }

        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }

    def _enrich_search_results(self, chroma_results: dict[str, Any], query: str) -> list[dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data, grouped by paper.

        With chunked indexing each paper has multiple docs in Chroma (one
        summary at ``#-1`` plus N fulltext chunks). A single query may
        surface several of them. We collapse those hits into one result per
        ``item_key``, preserve first-appearance order, and expose a sorted
        ``passages`` list so downstream consumers can pick the best chunk(s).

        This also dedupes the per-hit ``zotero_client.item(key)`` fetch —
        before this change, N chunks meant N API round-trips per paper.
        """
        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return []

        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []
        documents = chroma_results.get("documents", [[]])[0] if chroma_results.get("documents") else []
        metadatas = chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") else []

        def _item_key_of(doc_id: str, meta: dict[str, Any] | None) -> str:
            # Prefer metadata's item_key; fall back to parsing the doc_id.
            if meta and meta.get("item_key"):
                return meta["item_key"]
            return doc_id.split("#", 1)[0]

        def _chunk_idx_of(doc_id: str, meta: dict[str, Any] | None) -> int | None:
            if meta and "chunk_idx" in meta:
                try:
                    return int(meta["chunk_idx"])
                except (TypeError, ValueError):
                    pass
            if "#" in doc_id:
                _, _, tail = doc_id.partition("#")
                try:
                    return int(tail)
                except ValueError:
                    return None
            return None

        # Group hits by item_key, preserving first-appearance order.
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else 0.0
            sim = 1 - dist
            item_key = _item_key_of(doc_id, meta)
            chunk_idx = _chunk_idx_of(doc_id, meta)

            passage = {
                "chunk_idx": chunk_idx,
                "text": text,
                "similarity_score": sim,
                "metadata": meta,
            }

            if item_key not in groups:
                order.append(item_key)
                groups[item_key] = {
                    "item_key": item_key,
                    "best": passage,
                    "passages": [passage],
                }
            else:
                g = groups[item_key]
                g["passages"].append(passage)
                if sim > g["best"]["similarity_score"]:
                    g["best"] = passage

        # Fetch each Zotero item once.
        enriched: list[dict[str, Any]] = []
        for item_key in order:
            g = groups[item_key]
            # Sort passages by similarity desc (best first).
            passages_sorted = sorted(
                g["passages"], key=lambda p: p["similarity_score"], reverse=True
            )
            best = g["best"]

            try:
                zotero_item = self.zotero_client.item(item_key)
                enriched.append({
                    "item_key": item_key,
                    "similarity_score": best["similarity_score"],
                    "matched_text": best["text"],
                    "metadata": best["metadata"],
                    "zotero_item": zotero_item,
                    "passages": passages_sorted,
                    "query": query,
                })
            except Exception as e:
                logger.error(f"Error enriching result for item {item_key}: {e}")
                enriched.append({
                    "item_key": item_key,
                    "similarity_score": best["similarity_score"],
                    "matched_text": best["text"],
                    "metadata": best["metadata"],
                    "passages": passages_sorted,
                    "query": query,
                    "error": f"Could not fetch full item data: {e}",
                })

        return enriched

    def get_database_status(self) -> dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()

        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
        }

    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database.

        With chunking, one paper is spread across many Chroma records
        (``key#-1``, ``key#0``, …). Deletion goes through the metadata
        filter so all chunks — and any legacy bare-id record that still
        carries ``item_key`` metadata — vanish atomically.
        """
        try:
            self.chroma_client.delete_documents_where({"item_key": item_key})
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(config_path: str | None = None, db_path: str | None = None) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.

    Args:
        config_path: Path to configuration file
        db_path: Optional path to Zotero database (overrides config file)

    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(config_path=config_path, db_path=db_path)
