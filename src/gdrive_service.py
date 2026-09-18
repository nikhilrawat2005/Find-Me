import io
import os
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from src.config import (
    BASE_DIR,
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_TOKEN_PATH,
    GDRIVE_DIR
)

SCOPES = ['https://www.googleapis.com/auth/drive']
SUPPORTED_MIME_TYPES = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/webp': '.webp',
}

class GDriveService:
    def __init__(self, credentials_path: Optional[Path] = None):
        self.credentials_path = credentials_path or GDRIVE_CREDENTIALS_PATH
        self.auth_type = None
        self.account_email = None
        self._creds = None
        self._thread_local = threading.local()

    def get_service(self):
        """
        Returns a thread-local Google Drive service instance so concurrent threads
        never share the same underlying SSL socket / httplib2 connection.
        """
        if not hasattr(self._thread_local, "service") or self._thread_local.service is None:
            if not self.authenticate():
                raise RuntimeError("Google Drive credentials not authenticated.")
            # Build independent thread-safe client per worker thread
            self._thread_local.service = build('drive', 'v3', credentials=self._creds, cache_discovery=False)
        return self._thread_local.service

    @property
    def service(self):
        return self.get_service()

    def authenticate(self) -> bool:
        """
        Authenticates with Google Drive API using either:
        1. Service Account JSON file (recommended for permanent single-account integration)
        2. OAuth token / client credentials
        """
        if self._creds is not None:
            return True

        # Check for service account / credentials file
        if not self.credentials_path.exists():
            alt_paths = [
                BASE_DIR / "service_account.json",
                BASE_DIR / "data" / "credentials.json",
                BASE_DIR / "data" / "service_account.json"
            ]
            for p in alt_paths:
                if p.exists():
                    self.credentials_path = p
                    break

        if not self.credentials_path.exists():
            return False

        try:
            # First try loading as Service Account
            creds = service_account.Credentials.from_service_account_file(
                str(self.credentials_path), scopes=SCOPES
            )
            self._creds = creds
            self.auth_type = "service_account"
            self.account_email = creds.service_account_email
            return True
        except Exception as sa_err:
            # Try loading as OAuth credentials or saved token
            try:
                creds = None
                if GDRIVE_TOKEN_PATH.exists():
                    creds = Credentials.from_authorized_user_file(str(GDRIVE_TOKEN_PATH), SCOPES)
                
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        from google_auth_oauthlib.flow import InstalledAppFlow
                        flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                        creds = flow.run_local_server(port=0)
                    
                    with open(str(GDRIVE_TOKEN_PATH), 'w') as token_file:
                        token_file.write(creds.to_json())

                self._creds = creds
                self.auth_type = "oauth"
                return True
            except Exception as oauth_err:
                print(f"[GDrive Error] Auth failed: SA={sa_err}, OAuth={oauth_err}")
                return False

    @staticmethod
    def extract_folder_id(link_or_id: str) -> str:
        raw = link_or_id.strip()
        match = re.search(r'folders/([a-zA-Z0-9_-]+)', raw)
        if match:
            return match.group(1)
        match_id = re.search(r'[?&]id=([a-zA-Z0-9_-]+)', raw)
        if match_id:
            return match_id.group(1)
        return raw

    def check_connection(self) -> Dict[str, Any]:
        is_authenticated = self.authenticate()
        if not is_authenticated:
            return {
                "connected": False,
                "message": "credentials.json not found or authentication failed.",
                "credentials_path": str(self.credentials_path),
                "auth_type": None,
                "account_email": None
            }

        try:
            about = self.service.about().get(fields="user, storageQuota").execute()
            user_info = about.get("user", {})
            return {
                "connected": True,
                "auth_type": self.auth_type,
                "account_email": self.account_email or user_info.get("emailAddress", "Connected"),
                "display_name": user_info.get("displayName", "Drive User"),
                "storage_quota": about.get("storageQuota", {})
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "auth_type": self.auth_type,
                "account_email": self.account_email
            }

    def get_folder_details(self, folder_id: str) -> Dict[str, Any]:
        """
        Retrieves folder name and metadata from Google Drive.
        """
        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated.")
        
        cleaned_id = self.extract_folder_id(folder_id)
        try:
            folder = self.service.files().get(
                fileId=cleaned_id, 
                fields='id, name, mimeType, capabilities(canEdit, canAddChildren, canDelete, canTrashChildren)'
            ).execute()
            return folder
        except Exception as e:
            return {"id": cleaned_id, "name": f"Event_{cleaned_id[:6]}"}

    def check_folder_permission(self, folder_id: str) -> Dict[str, Any]:
        """
        Validates if the authenticated account has Editor permissions on the folder.
        """
        if not self.authenticate():
            return {"ok": False, "error": "Google Drive credentials not configured or auth failed."}
        
        cleaned_id = self.extract_folder_id(folder_id)
        try:
            folder = self.service.files().get(
                fileId=cleaned_id,
                fields='id, name, capabilities(canEdit, canAddChildren, canDelete, canTrashChildren)'
            ).execute()
            capabilities = folder.get("capabilities", {})
            can_edit = capabilities.get("canEdit", False) or capabilities.get("canAddChildren", False)
            
            return {
                "ok": True,
                "folder_id": cleaned_id,
                "folder_name": folder.get("name", "Unknown"),
                "can_edit": can_edit,
                "capabilities": capabilities,
                "account_email": self.account_email
            }
        except Exception as e:
            err_msg = str(e)
            if "404" in err_msg or "File not found" in err_msg:
                return {"ok": False, "error": f"Folder not found or not shared with {self.account_email or 'Service Account'}."}
            return {"ok": False, "error": err_msg}

    def list_folder_photos(self, folder_id: str) -> List[Dict[str, Any]]:
        """
        Lists all image files in the specified folder sorted by creation/name.
        """

        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated. Please provide credentials.json.")

        cleaned_id = self.extract_folder_id(folder_id)
        mime_query = " or ".join([f"mimeType = '{mime}'" for mime in SUPPORTED_MIME_TYPES.keys()])
        query = f"'{cleaned_id}' in parents and ({mime_query}) and trashed = false"

        results = []
        page_token = None

        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, size, md5Checksum, createdTime, modifiedTime, thumbnailLink, webViewLink)',
                pageToken=page_token,
                pageSize=100
            ).execute()

            files = response.get('files', [])
            results.extend(files)

            page_token = response.get('nextPageToken')
            if not page_token:
                break

        return results

    def get_photo_bytes(self, file_id: str, max_retries: int = 3) -> bytes:
        """
        Streams image file directly into memory (RAM) without saving to disk,
        with retry backoff for network stability.
        """
        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated.")

        import time
        last_err = None
        for attempt in range(max_retries):
            try:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                # 1MB chunk size is fast and resilient against transient drops
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                return fh.getvalue()
            except Exception as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise last_err

    def rename_file(self, file_id: str, new_name: str) -> Dict[str, Any]:
        """
        Renames a file directly on Google Drive.
        """
        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated.")

        body = {'name': new_name}
        return self.service.files().update(fileId=file_id, body=body, fields='id, name').execute()

    def delete_file(self, file_id: str, folder_id: Optional[str] = None) -> bool:
        """
        Safely removes a duplicate file from Google Drive:
        1. Tries moving file to Trash (trashed=True).
        2. If non-owner restriction prevents trashing, removes the file from this specific folder (removeParents).
        3. Falls back to delete if owner.
        """
        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated.")

        # Approach 1: Move to Trash (recommended & safer than hard delete)
        try:
            self.service.files().update(fileId=file_id, body={'trashed': True}).execute()
            return True
        except Exception:
            pass

        # Approach 2: If user is not owner, remove file from this folder (unlink)
        if folder_id:
            try:
                cleaned_folder = self.extract_folder_id(folder_id)
                self.service.files().update(
                    fileId=file_id,
                    removeParents=cleaned_folder,
                    fields='id, parents'
                ).execute()
                return True
            except Exception:
                pass

        # Approach 3: Hard delete
        try:
            self.service.files().delete(fileId=file_id).execute()
            return True
        except Exception as e:
            raise e


    def is_image_file(self, file_meta: Dict[str, Any]) -> bool:
        """
        Determines if a Google Drive file is an image based on MIME type or filename extension.
        Bypasses video MIME types and extensions.
        """
        mime = file_meta.get("mimeType", "").lower()
        name = file_meta.get("name", "").lower()

        # Instant skip for video types
        if mime.startswith("video/") or any(name.endswith(ext) for ext in [".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]):
            return False

        if mime in SUPPORTED_MIME_TYPES:
            return True

        # Fallback to extension check in case of custom mime types (e.g. iPhone HEIC or generic binary)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
        ext = Path(name).suffix.lower()
        return ext in valid_exts

    def crawl_folder_recursive(self, root_folder_id: str, current_path: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recursively walks Google Drive folder tree starting from root_folder_id.
        - Descends into every subfolder (e.g. Camera, iPhone, Day 1, Presentation, videos).
        - Skips video files instantly.
        - Identifies and retains any image files (even inside video folders).
        - Automatically computes `section_name` and `subfolder_path` for each photo.
        """
        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated.")

        if current_path is None:
            current_path = []

        cleaned_id = self.extract_folder_id(root_folder_id)
        results = []
        page_token = None

        # Query all non-trashed children (both folders and files) in this directory
        query = f"'{cleaned_id}' in parents and trashed = false"

        while True:
            response = self.service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType, size, md5Checksum, createdTime, modifiedTime, thumbnailLink, webViewLink)',
                pageToken=page_token,
                pageSize=100
            ).execute()

            items = response.get('files', [])
            for item in items:
                mime = item.get('mimeType', '')
                item_name = item.get('name', '').strip()

                if mime == 'application/vnd.google-apps.folder':
                    # Recurse into child directory
                    sub_path = current_path + [item_name]
                    sub_results = self.crawl_folder_recursive(item['id'], current_path=sub_path)
                    results.extend(sub_results)
                else:
                    # Check if file is an image (fast-skipping videos)
                    if self.is_image_file(item):
                        # Determine human-friendly section name
                        section_name = self._resolve_section_name(current_path)
                        subfolder_str = "/".join(current_path) if current_path else ""

                        item_copy = dict(item)
                        item_copy["section_name"] = section_name
                        item_copy["subfolder_path"] = subfolder_str
                        item_copy["parent_folder_id"] = cleaned_id
                        results.append(item_copy)

            page_token = response.get('nextPageToken')
            if not page_token:
                break

        return results

    @staticmethod
    def _resolve_section_name(path_parts: List[str]) -> str:
        """
        Resolves a clean section name from folder hierarchy.
        E.g.:
        ["Day 1", "Presentation", "photos"] -> "Presentation"
        ["Camera 1", "photos"] -> "Camera 1"
        ["iPhone 15"] -> "iPhone 15"
        ["Day 2", "Inauguration"] -> "Inauguration"
        [] -> "Photo"
        """
        if not path_parts:
            return "Photo"

        generic_names = {"photos", "images", "photo", "image", "raw", "pics", "pictures", "videos", "video", "dcim"}
        # Search backwards for the most descriptive folder name
        for part in reversed(path_parts):
            clean = part.strip()
            if clean.lower() not in generic_names and clean:
                # Replace underscores/dashes with spaces and title case
                return re.sub(r'[^\w\s-]', '', clean).replace(" ", "_")

        # Fallback to the first non-empty folder name
        return re.sub(r'[^\w\s-]', '', path_parts[0].strip()).replace(" ", "_")

    def batch_rename_recursive(self, crawled_photos: List[Dict[str, Any]], fresh_reset: bool = False) -> Dict[str, Any]:
        """
        Renames photos sequentially section-by-section directly on Google Drive.
        Format: <SectionName>_0001.jpg, <SectionName>_0002.jpg...
        Preserves existing valid section names when fresh_reset=False.
        """
        # Group crawled photos by parent folder
        folder_groups: Dict[str, List[Dict[str, Any]]] = {}
        for photo in crawled_photos:
            folder_id = photo.get("parent_folder_id")
            folder_groups.setdefault(folder_id, []).append(photo)

        total_renamed = 0
        renamed_details = []

        for folder_id, photos in folder_groups.items():
            if not photos:
                continue
            section = photos[0].get("section_name", "Photo")
            # Pattern matching: SectionName_0001.ext
            pattern = re.compile(rf"^{re.escape(section)}_(\d{{4}})\.(jpg|jpeg|png|webp|heic|heif)$", re.IGNORECASE)

            if fresh_reset:
                photos_sorted = sorted(photos, key=lambda x: (x.get("createdTime", ""), x.get("name", "")))
                for idx, f in enumerate(photos_sorted, start=1):
                    curr_name = f["name"]
                    ext = Path(curr_name).suffix.lower()
                    if not ext or ext not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
                        ext = ".jpg"

                    new_name = f"{section}_{idx:04d}{ext}"
                    if curr_name != new_name:
                        try:
                            self.rename_file(f["id"], new_name)
                            f["name"] = new_name
                            total_renamed += 1
                            renamed_details.append({"file_id": f["id"], "old_name": curr_name, "new_name": new_name})
                        except Exception as e:
                            print(f"[GDrive Error] Failed to rename {curr_name} to {new_name}: {e}")
            else:
                existing_numbers = set()
                unnamed_files = []

                for f in photos:
                    m = pattern.match(f["name"])
                    if m:
                        existing_numbers.add(int(m.group(1)))
                    else:
                        unnamed_files.append(f)

                next_idx = (max(existing_numbers) + 1) if existing_numbers else 1
                unnamed_sorted = sorted(unnamed_files, key=lambda x: (x.get("createdTime", ""), x.get("name", "")))

                for f in unnamed_sorted:
                    while next_idx in existing_numbers:
                        next_idx += 1

                    curr_name = f["name"]
                    ext = Path(curr_name).suffix.lower()
                    if not ext or ext not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
                        ext = ".jpg"

                    new_name = f"{section}_{next_idx:04d}{ext}"
                    try:
                        self.rename_file(f["id"], new_name)
                        f["name"] = new_name
                        existing_numbers.add(next_idx)
                        total_renamed += 1
                        renamed_details.append({"file_id": f["id"], "old_name": curr_name, "new_name": new_name})
                        next_idx += 1
                    except Exception as e:
                        print(f"[GDrive Error] Failed to rename {curr_name} to {new_name}: {e}")

        return {
            "total_files": len(crawled_photos),
            "renamed_count": total_renamed,
            "details": renamed_details
        }

    def find_and_clean_duplicates(self, folder_id: str, auto_delete: bool = False) -> Dict[str, Any]:
        """
        Scans all files across root and subfolders for duplicates (same md5 checksum only).
        Same-name files in DIFFERENT subfolders are NOT treated as duplicates — only truly
        identical binary content (md5) is removed.
        If auto_delete is True, keeps the earliest/primary uploaded copy and deletes duplicates.
        """
        files = self.crawl_folder_recursive(folder_id)
        hash_groups: Dict[str, List[Dict[str, Any]]] = {}

        for f in files:
            md5 = f.get("md5Checksum")
            if md5:
                hash_groups.setdefault(md5, []).append(f)

        duplicates = []
        deleted_count = 0
        seen_dup_ids = set()

        # Only deduplicate on identical file content (md5 hash)
        for md5, group in hash_groups.items():
            if len(group) > 1:
                group_sorted = sorted(group, key=lambda x: x.get("createdTime", x.get("modifiedTime", "")))
                keep = group_sorted[0]
                dupes = group_sorted[1:]
                for d in dupes:
                    if d["id"] not in seen_dup_ids:
                        seen_dup_ids.add(d["id"])
                        duplicates.append({
                            "reason": f"Identical file content (md5: {md5})",
                            "keep_id": keep["id"],
                            "keep_name": keep["name"],
                            "delete_id": d["id"],
                            "delete_name": d["name"],
                            "size": d.get("size", 0)
                        })
                        if auto_delete:
                            try:
                                self.delete_file(d["id"], folder_id=d.get("parent_folder_id", folder_id))
                                deleted_count += 1
                            except Exception as del_err:
                                print(f"[GDrive Error] Failed to delete duplicate {d['id']}: {del_err}")

        return {
            "total_files_scanned": len(files),
            "duplicates_found": len(duplicates),
            "duplicates_deleted": deleted_count,
            "details": duplicates
        }

    def batch_rename_folder(self, folder_id: str, prefix: str = "photo", fresh_reset: bool = False) -> Dict[str, Any]:
        """
        Sequential renamer supporting both single folder and deep recursive folder structures.
        """
        crawled = self.crawl_folder_recursive(folder_id)
        return self.batch_rename_recursive(crawled, fresh_reset=fresh_reset)

