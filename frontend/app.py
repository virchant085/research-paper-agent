"""Streamlit UI for the Research Paper Agent.

Standalone frontend that talks to the FastAPI backend over HTTP with `requests`.
Run with:  streamlit run frontend/app.py

Configuration:
    API_BASE  (env var)  -> base URL of the backend API. Defaults to
                            "http://localhost:8000". Editable in the sidebar.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

DEFAULT_API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
REQUEST_TIMEOUT = 120  # seconds; ingestion / agent calls can be slow.

CONNECTION_HINT = (
    "Could not reach the API. Make sure the backend is running "
    "(e.g. `python -m backend.main` or `uvicorn backend.main:app`) "
    "and that the API base URL above is correct."
)


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def _url(api_base: str, path: str) -> str:
    """Join the API base and a path, avoiding double slashes."""
    return api_base.rstrip("/") + "/" + path.lstrip("/")


def api_get(api_base: str, path: str, **kwargs):
    """GET helper returning the parsed JSON body (raises on HTTP/connection error)."""
    resp = requests.get(_url(api_base, path), timeout=REQUEST_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_post(api_base: str, path: str, **kwargs):
    """POST helper returning the parsed JSON body (raises on HTTP/connection error)."""
    resp = requests.post(_url(api_base, path), timeout=REQUEST_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp.json()


def api_delete(api_base: str, path: str, **kwargs):
    """DELETE helper returning the parsed JSON body (raises on HTTP/connection error)."""
    resp = requests.delete(_url(api_base, path), timeout=REQUEST_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp.json()


def fetch_papers(api_base: str) -> list[dict]:
    """Fetch the list of paper cards. Returns [] and shows an error on failure."""
    try:
        return api_get(api_base, "/papers")
    except requests.exceptions.ConnectionError:
        st.error(CONNECTION_HINT)
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to load papers: {exc}")
    return []


def _paper_label(card: dict) -> str:
    """Human-friendly label for a paper card in select widgets."""
    title = (card.get("title") or "").strip() or "(untitled)"
    pid = card.get("paper_id", "")
    return f"{title}  [{pid}]"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> str:
    """Render the sidebar (API base + health) and return the chosen API base."""
    st.sidebar.title("Research Paper Agent")
    api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backend health")
    try:
        health = api_get(api_base, "/health")
        provider = health.get("provider", "unknown")
        status = health.get("status", "unknown")
        st.sidebar.success(f"status: {status}  \nprovider: {provider}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("API unreachable")
        st.sidebar.caption(CONNECTION_HINT)
    except requests.exceptions.RequestException as exc:
        st.sidebar.error(f"Health check failed: {exc}")

    return api_base


# --------------------------------------------------------------------------- #
# Papers tab
# --------------------------------------------------------------------------- #
def render_papers_tab(api_base: str) -> None:
    """Upload PDFs / arXiv papers and manage the ingested library."""
    st.header("Papers")

    # --- PDF upload ------------------------------------------------------- #
    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("PDF", type=["pdf"])
    if st.button("Ingest PDF", disabled=uploaded is None):
        if uploaded is None:
            st.warning("Choose a PDF file first.")
        else:
            with st.spinner("Uploading and ingesting PDF..."):
                try:
                    files = {
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            "application/pdf",
                        )
                    }
                    result = api_post(api_base, "/papers/upload", files=files)
                    _show_ingest_result(result)
                except requests.exceptions.ConnectionError:
                    st.error(CONNECTION_HINT)
                except requests.exceptions.RequestException as exc:
                    st.error(f"Upload failed: {exc}")

    st.markdown("---")

    # --- arXiv ingest ----------------------------------------------------- #
    st.subheader("Add from arXiv")
    arxiv_url = st.text_input(
        "arXiv URL or ID",
        placeholder="https://arxiv.org/abs/2301.00001  or  2301.00001",
    )
    if st.button("Ingest arXiv", disabled=not arxiv_url.strip()):
        with st.spinner("Downloading and ingesting from arXiv..."):
            try:
                result = api_post(
                    api_base, "/papers/arxiv", json={"url": arxiv_url.strip()}
                )
                _show_ingest_result(result)
            except requests.exceptions.ConnectionError:
                st.error(CONNECTION_HINT)
            except requests.exceptions.RequestException as exc:
                st.error(f"arXiv ingest failed: {exc}")

    st.markdown("---")

    # --- Library listing -------------------------------------------------- #
    st.subheader("Ingested papers")
    papers = fetch_papers(api_base)
    if not papers:
        st.info("No papers ingested yet.")
        return

    for card in papers:
        pid = card.get("paper_id", "")
        with st.expander(_paper_label(card)):
            _render_card_fields(card)
            if st.button("Delete", key=f"delete_{pid}"):
                try:
                    api_delete(api_base, f"/papers/{pid}")
                    st.success(f"Deleted {pid}.")
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error(CONNECTION_HINT)
                except requests.exceptions.RequestException as exc:
                    st.error(f"Delete failed: {exc}")


def _show_ingest_result(result: dict) -> None:
    """Render an IngestResponse dict as success/error."""
    status = (result.get("status") or "").lower()
    title = result.get("title", "")
    pid = result.get("paper_id", "")
    message = result.get("message", "")
    if status == "ok":
        st.success(f"Ingested: {title}  [{pid}]")
    else:
        st.error(f"Ingestion failed: {message or 'unknown error'}")


def _render_card_fields(card: dict) -> None:
    """Show the fields of a PaperCard inside an expander."""
    authors = card.get("authors") or []
    authors_str = ", ".join(authors) if authors else "-"
    year = card.get("year")
    st.markdown(f"**Paper ID:** {card.get('paper_id', '')}")
    st.markdown(f"**Authors:** {authors_str}")
    st.markdown(f"**Year:** {year if year is not None else '-'}")
    st.markdown(f"**Source:** {card.get('source', '') or '-'}")
    st.markdown(f"**Problem:** {card.get('problem', '') or '-'}")
    st.markdown(f"**Method:** {card.get('method', '') or '-'}")
    st.markdown(f"**Dataset:** {card.get('dataset', '') or '-'}")
    st.markdown(f"**Contribution:** {card.get('contribution', '') or '-'}")
    st.markdown(f"**Limitation:** {card.get('limitation', '') or '-'}")


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
def render_chat_tab(api_base: str) -> None:
    """Ask questions of the research agent, optionally scoped to papers."""
    st.header("Chat")

    papers = fetch_papers(api_base)
    label_to_id = {_paper_label(c): c.get("paper_id", "") for c in papers}

    question = st.text_area(
        "Question",
        placeholder="e.g. What datasets do these papers use, and how do their methods differ?",
        height=120,
    )
    selected_labels = st.multiselect(
        "Restrict to papers (optional)",
        options=list(label_to_id.keys()),
    )

    if st.button("Ask", disabled=not question.strip()):
        paper_ids = [label_to_id[lbl] for lbl in selected_labels] or None
        payload = {"question": question.strip(), "paper_ids": paper_ids}
        with st.spinner("Thinking..."):
            try:
                result = api_post(api_base, "/query", json=payload)
            except requests.exceptions.ConnectionError:
                st.error(CONNECTION_HINT)
                return
            except requests.exceptions.RequestException as exc:
                st.error(f"Query failed: {exc}")
                return

        answer = result.get("answer", "") or "(no answer returned)"
        st.subheader("Answer")
        st.markdown(answer)

        steps = result.get("steps") or []
        if steps:
            st.subheader("Tool steps")
            for i, step in enumerate(steps, start=1):
                tool = step.get("tool", "")
                with st.expander(f"Step {i}: {tool}"):
                    st.markdown("**Arguments:**")
                    st.json(step.get("arguments", {}))
                    preview = step.get("result_preview", "")
                    if preview:
                        st.markdown("**Result preview:**")
                        st.code(preview)


# --------------------------------------------------------------------------- #
# Library tab
# --------------------------------------------------------------------------- #
def render_library_tab(api_base: str) -> None:
    """Literature table, pairwise comparison, and export."""
    st.header("Library")

    papers = fetch_papers(api_base)
    if not papers:
        st.info("No papers ingested yet.")
        return

    label_to_id = {_paper_label(c): c.get("paper_id", "") for c in papers}

    # --- Literature table ------------------------------------------------- #
    st.subheader("Literature table")
    try:
        table = api_post(api_base, "/table", json={"paper_ids": None})
        md = table.get("markdown", "")
        if md.strip():
            st.markdown(md)
        else:
            st.info("Literature table is empty.")
    except requests.exceptions.ConnectionError:
        st.error(CONNECTION_HINT)
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to build literature table: {exc}")

    st.markdown("---")

    # --- Compare ---------------------------------------------------------- #
    st.subheader("Compare papers")
    compare_labels = st.multiselect(
        "Select papers to compare",
        options=list(label_to_id.keys()),
        key="compare_select",
    )
    if st.button("Compare", disabled=len(compare_labels) < 2):
        paper_ids = [label_to_id[lbl] for lbl in compare_labels]
        with st.spinner("Comparing..."):
            try:
                result = api_post(
                    api_base, "/compare", json={"paper_ids": paper_ids}
                )
                st.markdown(result.get("markdown", "") or "(no comparison returned)")
            except requests.exceptions.ConnectionError:
                st.error(CONNECTION_HINT)
            except requests.exceptions.RequestException as exc:
                st.error(f"Compare failed: {exc}")
    elif len(compare_labels) == 1:
        st.caption("Select at least two papers to compare.")

    st.markdown("---")

    # --- Export ----------------------------------------------------------- #
    st.subheader("Export")
    export_format = st.radio("Format", options=["markdown", "csv"], horizontal=True)
    if st.button("Export"):
        with st.spinner("Exporting..."):
            try:
                result = api_post(
                    api_base,
                    "/export",
                    json={"format": export_format, "paper_ids": None},
                )
            except requests.exceptions.ConnectionError:
                st.error(CONNECTION_HINT)
                return
            except requests.exceptions.RequestException as exc:
                st.error(f"Export failed: {exc}")
                return

        path = result.get("path", "")
        fmt = result.get("format", export_format)
        st.success(f"Exported to: {path}")
        _offer_download(path, fmt)


def _offer_download(path: str, fmt: str) -> None:
    """Read the exported file from disk and offer a download button.

    The frontend usually runs on the same host as the backend for this
    prototype, so the returned path is a local filesystem path.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        st.warning(
            f"Export written on the server at:\n\n`{path}`\n\n"
            f"(Could not read it locally for download: {exc})"
        )
        return

    filename = os.path.basename(path) or f"export.{'csv' if fmt == 'csv' else 'md'}"
    mime = "text/csv" if fmt == "csv" else "text/markdown"
    st.download_button(
        "Download export",
        data=data,
        file_name=filename,
        mime=mime,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Research Paper Agent", layout="wide")
    api_base = render_sidebar()

    papers_tab, chat_tab, library_tab = st.tabs(["Papers", "Chat", "Library"])
    with papers_tab:
        render_papers_tab(api_base)
    with chat_tab:
        render_chat_tab(api_base)
    with library_tab:
        render_library_tab(api_base)


if __name__ == "__main__":
    main()
