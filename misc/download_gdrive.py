"""
download_gdrive.py — Download a Google Drive folder (or file) to a local path.

Usage
-----
    python download_gdrive.py <gdrive_url_or_id> <destination_dir> [--unzip]

Examples

--------
    # Download a shared folder by URL
    python download_gdrive.py \
        "https://drive.google.com/drive/folders/1AbCdEf..." \
        ./smell_ts_dataset

    # Download a shared file by ID
    python download_gdrive.py \
        1AbCdEf... \
        ./data/myfile.zip --unzip

    # Download a private folder using a service-account JSON key
    python download_gdrive.py \
        1AbCdEf... \
        ./data --service-account credentials.json

Requirements
------------
    pip install gdown tqdm

For private files/folders (service-account auth):
    pip install google-api-python-client google-auth

How to get the folder/file ID
------------------------------
    From a URL like:
        https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUv
    The ID is the last path segment:  1AbCdEfGhIjKlMnOpQrStUv

    For a file:
        https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUv/view
    The ID is between /d/ and /view:  1AbCdEfGhIjKlMnOpQrStUv
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_id(url_or_id: str) -> str:
    """Extract the Drive file/folder ID from a URL, or return as-is."""
    # Folder URL: .../folders/<ID>
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    # File URL: /d/<ID>/
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    # id= query param
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if m:
        return m.group(1)
    # Assume it's already a raw ID
    return url_or_id


def is_folder_url(url_or_id: str) -> bool:
    return "/folders/" in url_or_id


# ---------------------------------------------------------------------------
# Download via gdown (public / link-shared files and folders)
# ---------------------------------------------------------------------------

def download_with_gdown(
    url_or_id: str,
    dest: Path,
    is_folder: bool,
) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed.  Run: pip install gdown")

    dest.mkdir(parents=True, exist_ok=True)

    drive_id = extract_id(url_or_id)

    if is_folder:
        folder_url = f"https://drive.google.com/drive/folders/{drive_id}"
        print(f"Downloading folder {drive_id} → {dest}")
        gdown.download_folder(
            url=folder_url,
            output=str(dest),
            quiet=False,
            use_cookies=False,
        )
    else:
        file_url = f"https://drive.google.com/uc?id={drive_id}"
        # Guess output filename; gdown will use the Drive filename if output is a dir
        print(f"Downloading file {drive_id} → {dest}")
        gdown.download(
            url=file_url,
            output=str(dest) + "/",   # trailing slash → treat as directory
            quiet=False,
        )


# ---------------------------------------------------------------------------
# Download via Google Drive API (private files, service-account auth)
# ---------------------------------------------------------------------------

def download_with_api(
    drive_id: str,
    dest: Path,
    service_account_json: str,
    is_folder: bool,
) -> None:
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import io
    except ImportError:
        sys.exit(
            "Google API client not installed.\n"
            "Run: pip install google-api-python-client google-auth"
        )

    creds = service_account.Credentials.from_service_account_file(
        service_account_json,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    service = build("drive", "v3", credentials=creds)

    dest.mkdir(parents=True, exist_ok=True)

    if is_folder:
        _download_folder_api(service, drive_id, dest)
    else:
        _download_file_api(service, drive_id, dest)


def _download_file_api(service, file_id: str, dest_dir: Path) -> None:
    meta = service.files().get(fileId=file_id, fields="name,mimeType").execute()
    name = meta["name"]
    out_path = dest_dir / name
    print(f"  ↓ {name}")

    request = service.files().get_media(fileId=file_id)
    import io
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"    {int(status.progress() * 100)}%", end="\r")
    print()
    out_path.write_bytes(buf.getvalue())


def _download_folder_api(service, folder_id: str, dest_dir: Path) -> None:
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None
    while True:
        resp = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
        ).execute()

        for item in resp.get("files", []):
            if item["mimeType"] == "application/vnd.google-apps.folder":
                sub_dir = dest_dir / item["name"]
                sub_dir.mkdir(exist_ok=True)
                _download_folder_api(service, item["id"], sub_dir)
            else:
                _download_file_api(service, item["id"], dest_dir)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break


# ---------------------------------------------------------------------------
# Unzip helper
# ---------------------------------------------------------------------------

def maybe_unzip(dest: Path) -> None:
    zips = list(dest.glob("*.zip"))
    if not zips:
        print("No .zip files found to extract.")
        return
    for z in zips:
        print(f"Extracting {z.name} …")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)
        z.unlink()
        print(f"  Extracted and removed {z.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Google Drive folder or file to a local directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help="Google Drive share URL or file/folder ID.",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Local directory to download into.",
    )
    parser.add_argument(
        "--folder",
        action="store_true",
        default=None,
        help=(
            "Treat source as a folder ID (auto-detected from URL; "
            "needed when passing a raw ID for a folder)."
        ),
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="After download, extract any .zip files found in the destination.",
    )
    parser.add_argument(
        "--service-account",
        metavar="CREDENTIALS_JSON",
        default=None,
        help=(
            "Path to a Google service-account JSON key file. "
            "Use this for private/non-shared Drive files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect whether it's a folder link
    if args.folder is None:
        is_folder = is_folder_url(args.source)
    else:
        is_folder = args.folder

    dest = args.destination.expanduser().resolve()

    if args.service_account:
        drive_id = extract_id(args.source)
        download_with_api(
            drive_id=drive_id,
            dest=dest,
            service_account_json=args.service_account,
            is_folder=is_folder,
        )
    else:
        download_with_gdown(
            url_or_id=args.source,
            dest=dest,
            is_folder=is_folder,
        )

    if args.unzip:
        maybe_unzip(dest)

    print(f"\nDone. Files saved to: {dest}")


if __name__ == "__main__":
    main()
