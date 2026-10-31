"""Windows-native HTTPS download fallback for make-usbroot.ps1.

Uses Python's OpenSSL-backed urllib rather than Windows Schannel. The caller
verifies the pinned SHA-256 before using the downloaded file.
"""

import sys
import os
import urllib.request


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: download-pinned.py URL OUTPUT", file=sys.stderr)
        return 2

    url, output = sys.argv[1:]
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    # CachyOS's CDN rejects urllib's default Python-urllib user agent (403).
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected download status: {response.status}")
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        next_report = 16 * 1024 * 1024
        with open(output, "wb") as target:
            while chunk := response.read(64 * 1024):
                target.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        print(f"[download] {downloaded}/{total} bytes", flush=True)
                    else:
                        print(f"[download] {downloaded} bytes", flush=True)
                    next_report += 16 * 1024 * 1024
    return 0


if __name__ == "__main__":
    sys.exit(main())
