"""Read single members of a large remote zip over HTTP range requests (no full download).

    with RemoteZip(url) as z:
        data = z.read("ai_001_001/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg")
"""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small read-ahead block cache."""

    def __init__(self, url: str, *, block: int = 1 << 20, retries: int = 5):
        self.url, self.block, self.retries = url, block, retries
        request = urllib.request.Request(url, method="HEAD")
        with self._open(request) as response:
            self.size = int(response.headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}

    def _open(self, request):
        for attempt in range(self.retries):
            try:
                return urllib.request.urlopen(request, timeout=60)
            except OSError:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def _fetch(self, start: int, end: int) -> bytes:  # inclusive end
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(self.retries):
            try:
                with self._open(request) as response:
                    data = response.read()
                if len(data) == end - start + 1:
                    return data
            except OSError:
                pass
            time.sleep(2 ** attempt)
        raise OSError(f"range {start}-{end} of {self.url} failed")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > self.block:  # large member: fetch directly
            data = self._fetch(self.pos, self.pos + n - 1)
        else:
            first, last = self.pos // self.block, (self.pos + n - 1) // self.block
            chunks = []
            for b in range(first, last + 1):
                if b not in self.cache:
                    if len(self.cache) > 64:
                        self.cache.clear()
                    self.cache[b] = self._fetch(b * self.block, min(self.size, (b + 1) * self.block) - 1)
                chunks.append(self.cache[b])
            joined = b"".join(chunks)
            start = self.pos - first * self.block
            data = joined[start:start + n]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class RemoteZip(zipfile.ZipFile):
    def __init__(self, url: str):
        super().__init__(HTTPRangeFile(url))
