"""
Extracts only the needed images from a large remote zip, using HTTP Range requests,
instead of downloading the whole archive. Written for the skin-tone audit datasets
(Fitzpatrick17k-C images zip on Zenodo, 1.4 GB; DermaCon-IN DATASET_0/1.zip on Harvard
Dataverse, 3.6 GB) on a slow connection, where only a fraction of each archive's images
map onto the Stage B classes.

Only the zip's central directory is read through zipfile; each wanted member is then
fetched with its own Range request (local header + data), several in parallel, since
the servers throttle per connection. If a partial local copy of the archive exists
(e.g. an interrupted curl), members it fully covers are read from disk instead.

Usage (as a module):
    fetch_members(url, wanted_basenames, out_dir, partial_local=None)
"""
import http.client
import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Harvard Dataverse returns 403 to the default Python-urllib User-Agent.
UA = {"User-Agent": "Mozilla/5.0 (research dataset download)"}
WORKERS = 8  # Zenodo answers 429 above roughly this many parallel requests
HEADER_SLACK = 1024  # local header = 30 bytes + name + extra field; extra is not in the CD


def resolve(url):
    """Dataverse answers with a short-lived signed S3 redirect; returns the final URL."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", **UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = r.headers.get("Content-Range", "").split("/")[-1]
        return r.geturl(), int(total)


def get_range(url, start, end, tries=8):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", **UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError,
                OSError) as e:
            last = e
            throttled = isinstance(e, urllib.error.HTTPError) and e.code == 429
            time.sleep((20 if throttled else 2) * (attempt + 1))
    raise last


class RemoteFile(io.RawIOBase):
    """Minimal seekable file over Range requests -- only used to read the central
    directory at the end of the archive (a few requests)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data = get_range(self.url, self.pos, self.pos + n - 1)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def decode_member(blob, info):
    """blob starts at the member's local file header."""
    sig, *_ = struct.unpack("<I", blob[:4])
    if sig != 0x04034B50:
        raise ValueError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    data = blob[start:start + info.compress_size]
    if len(data) < info.compress_size:
        raise ValueError(f"short read for {info.filename}")
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise ValueError(f"unsupported compression {info.compress_type}")
    if zlib.crc32(out) & 0xFFFFFFFF != info.CRC:
        raise ValueError(f"CRC mismatch for {info.filename}")
    return out


def fetch_members(url, wanted, out_dir, partial_local=None):
    """Extract zip members whose basename is in `wanted` into out_dir (flat)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_url, size = resolve(url)
    zf = zipfile.ZipFile(RemoteFile(final_url, size))
    members = [i for i in zf.infolist() if Path(i.filename).name in wanted and not i.is_dir()]
    todo = [i for i in members if not (out_dir / Path(i.filename).name).exists()]
    local = Path(partial_local) if partial_local and Path(partial_local).exists() else None
    local_size = local.stat().st_size if local else 0
    print(f"{url}: {len(zf.infolist())} members, {len(members)} wanted, {len(todo)} to fetch "
          f"({sum(i.compress_size for i in todo) / 1e6:.0f} MB)", flush=True)

    current = [final_url]

    def one(info):
        end = info.header_offset + 30 + len(info.filename.encode()) + HEADER_SLACK + info.compress_size
        end = min(end, size) - 1
        if end < local_size:
            with open(local, "rb") as fh:
                fh.seek(info.header_offset)
                blob = fh.read(end - info.header_offset + 1)
        else:
            try:
                blob = get_range(current[0], info.header_offset, end, tries=3)
            except urllib.error.HTTPError:
                # signed S3 URLs expire after an hour -- re-resolve and retry
                current[0] = resolve(url)[0]
                blob = get_range(current[0], info.header_offset, end)
        (out_dir / Path(info.filename).name).write_bytes(decode_member(blob, info))

    failed = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(one, i): i for i in todo}
        for k, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
            except Exception as e:
                failed += 1
                print(f"  FAILED {futures[f].filename}: {e}", flush=True)
            if k % 200 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    print(f"  done: {len(todo) - failed} extracted, {failed} failed", flush=True)
    return members
