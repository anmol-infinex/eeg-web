"""Zero-dependency HTTP REST Client for E2E Testing.

Uses Python standard library `urllib.request` and `http.client` to execute
REST API operations against the Seizure-Sentry backend.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class HttpResponse:
    """Encapsulates an HTTP response with tuple-unpacking support.

    Can be unpacked as:
        status_code, json_data, content = client.get("/health")
    Or used as an object:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
    """

    def __init__(
        self,
        status_code: int,
        json_data: Any,
        content: bytes,
        headers: Dict[str, str],
        url: str = "",
    ):
        self.status_code = status_code
        self.json_data = json_data
        self.content = content
        self.headers = headers
        self.url = url

    @property
    def ok(self) -> bool:
        """True if status code is in the 2xx range."""
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        """Decodes content to string using UTF-8."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Returns parsed JSON data or raises ValueError if body is not JSON."""
        if self.json_data is not None:
            return self.json_data
        try:
            return json.loads(self.text)
        except Exception as e:
            raise ValueError(f"Response body is not valid JSON: {self.text[:200]}") from e

    def __iter__(self) -> Iterator[Any]:
        """Support unpacking as (status_code, json_data, content)."""
        return iter((self.status_code, self.json_data, self.content))

    def __getitem__(self, index: int) -> Any:
        return (self.status_code, self.json_data, self.content)[index]

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"<HttpResponse [{self.status_code}] bytes={len(self.content)}>"


class TestHttpClient:
    """Synchronous HTTP REST client with authentication and error handling."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        access_key: Optional[str] = None,
        timeout: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.access_key = access_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        json_data: Optional[Union[Dict, List, Any]] = None,
        headers: Optional[Dict[str, Optional[str]]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        """Send an HTTP request and return an HttpResponse.

        Does not raise for 4xx/5xx responses; instead captures status and body.
        """
        path = path.lstrip("/")
        url = f"{self.base_url}/{path}" if path else self.base_url

        if params:
            query = urllib.parse.urlencode(params)
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{query}"

        req_headers: Dict[str, str] = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "SeizureSentry-E2E-TestClient/1.0",
        }

        # Inject default access key unless explicitly suppressed
        if self.access_key:
            req_headers["X-Access-Key"] = self.access_key

        if headers:
            for k, v in headers.items():
                if v is None:
                    # Allow suppressing default headers by passing None
                    req_headers.pop(k, None)
                else:
                    req_headers[k] = str(v)

        req_body: Optional[bytes] = body
        if json_data is not None:
            req_body = json.dumps(json_data).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        req = urllib.request.Request(
            url=url,
            data=req_body,
            headers=req_headers,
            method=method.upper(),
        )

        effective_timeout = timeout if timeout is not None else self.timeout

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                status_code = resp.status
                raw_bytes = resp.read()
                resp_headers = dict(resp.getheaders())
        except urllib.error.HTTPError as http_err:
            status_code = http_err.code
            raw_bytes = http_err.read()
            resp_headers = dict(http_err.headers.items()) if http_err.headers else {}
        except urllib.error.URLError as url_err:
            # Re-raise connection/DNS errors so tests know backend was unreachable
            raise ConnectionError(f"HTTP connection to {url} failed: {url_err.reason}") from url_err

        # Attempt JSON parsing
        parsed_json = None
        content_type = resp_headers.get("content-type", resp_headers.get("Content-Type", ""))
        if "application/json" in content_type or raw_bytes.strip().startswith((b"{", b"[")):
            try:
                parsed_json = json.loads(raw_bytes.decode("utf-8"))
            except Exception:
                parsed_json = None

        return HttpResponse(
            status_code=status_code,
            json_data=parsed_json,
            content=raw_bytes,
            headers=resp_headers,
            url=url,
        )

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Optional[str]]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        return self.request("GET", path, headers=headers, params=params, timeout=timeout)

    def post(
        self,
        path: str,
        json_data: Optional[Union[Dict, List, Any]] = None,
        headers: Optional[Dict[str, Optional[str]]] = None,
        body: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        return self.request(
            "POST", path, json_data=json_data, headers=headers, body=body, timeout=timeout
        )

    def put(
        self,
        path: str,
        json_data: Optional[Union[Dict, List, Any]] = None,
        headers: Optional[Dict[str, Optional[str]]] = None,
        body: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        return self.request(
            "PUT", path, json_data=json_data, headers=headers, body=body, timeout=timeout
        )

    def delete(
        self,
        path: str,
        headers: Optional[Dict[str, Optional[str]]] = None,
        timeout: Optional[float] = None,
    ) -> HttpResponse:
        return self.request("DELETE", path, headers=headers, timeout=timeout)
