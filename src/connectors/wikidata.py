"""Minimal MediaWiki Action API clients for Wikidata and Wikipedia (read-only)."""
from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any

import requests

from src.core.exceptions import UpstreamError

_BATCH = 50  # most Action API modules accept at most 50 ids or titles per anonymous request


def _batches(items: Iterable[str]) -> Iterator[list[str]]:
    ordered = sorted(set(items))
    for start in range(0, len(ordered), _BATCH):
        yield ordered[start : start + _BATCH]


class ActionApiClient:
    """GET requests against one `api.php`; anonymous clients get about 10 requests per minute."""

    def __init__(self, endpoint: str, user_agent: str, interval_s: float, timeout_s: float) -> None:
        self._endpoint = endpoint
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._last_call = 0.0

    def get(self, params: dict[str, Any]) -> dict[str, Any]:
        """One request, spaced from the previous one; an API error payload raises UpstreamError."""
        wait = self._interval_s - (time.monotonic() - self._last_call)
        if self._last_call and wait > 0:
            time.sleep(wait)
        response = self._session.get(
            self._endpoint, timeout=self._timeout_s, params={**params, "format": "json", "formatversion": 2}
        )
        self._last_call = time.monotonic()
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        if "error" in payload:
            raise UpstreamError(f"{params.get('action')} -> {payload['error'].get('code')} from {self._endpoint}")
        return payload

    def paged(self, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Follow `continue` tokens, yielding each response."""
        cont: dict[str, Any] = {}
        while True:
            payload = self.get({**params, **cont})
            yield payload
            if "continue" not in payload:
                return
            cont = payload["continue"]


class WikidataClient(ActionApiClient):
    """Labels and sitelinks of Wikidata items."""

    def labels(self, qids: Iterable[str], languages: tuple[str, ...]) -> dict[str, dict[str, str]]:
        """Labels per item in the given languages; items without any of them are left out."""
        result: dict[str, dict[str, str]] = {}
        for batch in _batches(qids):
            payload = self.get({"action": "wbgetentities", "ids": "|".join(batch), "props": "labels",
                                "languages": "|".join(languages)})
            for qid, entity in payload.get("entities", {}).items():
                found = {lang: value["value"] for lang, value in (entity.get("labels") or {}).items()}
                if found:
                    result[qid] = found
        return result

    def claim_items(self, qids: Iterable[str], prop: str) -> dict[str, list[str]]:
        """Item ids each entity points to through one property (deprecated claims skipped)."""
        result: dict[str, list[str]] = {}
        for batch in _batches(qids):
            payload = self.get({"action": "wbgetentities", "ids": "|".join(batch), "props": "claims"})
            for qid, entity in payload.get("entities", {}).items():
                values = [
                    claim["mainsnak"]["datavalue"]["value"]["id"]
                    for claim in (entity.get("claims") or {}).get(prop, [])
                    if claim.get("rank") != "deprecated" and claim["mainsnak"].get("snaktype") == "value"
                ]
                if values:
                    result[qid] = values
        return result

    def sitelink_titles(self, qids: Iterable[str], site: str) -> dict[str, str]:
        """Article title of each item on one wiki (e.g. `viwiki`); items without one are left out."""
        result: dict[str, str] = {}
        for batch in _batches(qids):
            payload = self.get({"action": "wbgetentities", "ids": "|".join(batch), "props": "sitelinks",
                                "sitefilter": site})
            for qid, entity in payload.get("entities", {}).items():
                link = (entity.get("sitelinks") or {}).get(site)
                if link:
                    result[qid] = link["title"]
        return result


class WikipediaClient(ActionApiClient):
    """Article source text and article -> Wikidata item mapping of one Wikipedia."""

    def wikitext(self, titles: Iterable[str]) -> dict[str, str]:
        """Current wikitext keyed by the requested title."""
        result: dict[str, str] = {}
        for batch in _batches(titles):
            for payload in self.paged({"action": "query", "titles": "|".join(batch), "prop": "revisions",
                                       "rvprop": "content", "rvslots": "main"}):
                for page in payload.get("query", {}).get("pages", []):
                    revisions = page.get("revisions")
                    if revisions:
                        result[page["title"]] = revisions[0]["slots"]["main"]["content"]
        return result

    def page_items(self, titles: Iterable[str]) -> dict[str, str]:
        """Wikidata id of each requested title, following redirects; pages without an item are left out."""
        result: dict[str, str] = {}
        for batch in _batches(titles):
            for payload in self.paged({"action": "query", "titles": "|".join(batch), "prop": "pageprops",
                                       "ppprop": "wikibase_item", "redirects": 1}):
                query = payload.get("query", {})
                source = {step["to"]: step["from"] for step in query.get("normalized", [])}
                for step in query.get("redirects", []):
                    source[step["to"]] = source.get(step["from"], step["from"])
                for page in query.get("pages", []):
                    item = page.get("pageprops", {}).get("wikibase_item")
                    if item:
                        result[source.get(page["title"], page["title"])] = item
        return result
