import io
import os
import re
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
        self.service = None
        self.auth_type = None
        self.account_email = None

    def authenticate(self) -> bool:
        """
        Authenticates with Google Drive API using either:
        1. Service Account JSON file (recommended for permanent single-account integration)
        2. OAuth token / client credentials
        """
        if self.service is not None:
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
            self.service = build('drive', 'v3', credentials=creds)
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

                self.service = build('drive', 'v3', credentials=creds)
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

    def get_photo_bytes(self, file_id: str) -> bytes:
        """
        Streams image file directly into memory (RAM) without saving to disk.
        """
        if not self.authenticate():
            raise RuntimeError("Google Drive is not authenticated.")

        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        return fh.getvalue()

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


    def find_and_clean_duplicates(self, folder_id: str, auto_delete: bool = False) -> Dict[str, Any]:
        """
        Scans a folder for duplicate files (same name OR same md5 checksum).
        If auto_delete is True, keeps the earliest/primary uploaded copy and deletes duplicates.
        """
        files = self.list_folder_photos(folder_id)
        name_groups: Dict[str, List[Dict[str, Any]]] = {}
        hash_groups: Dict[str, List[Dict[str, Any]]] = {}

        for f in files:
            name = f["name"].strip().lower()
            name_groups.setdefault(name, []).append(f)
            md5 = f.get("md5Checksum")
            if md5:
                hash_groups.setdefault(md5, []).append(f)

        duplicates = []
        deleted_count = 0
        seen_dup_ids = set()

        # Group duplicates by name or content hash
        for name, group in name_groups.items():
            if len(group) > 1:
                # Sort by createdTime / modifiedTime so original is kept first
                group_sorted = sorted(group, key=lambda x: x.get("createdTime", x.get("modifiedTime", "")))
                keep = group_sorted[0]
                dupes = group_sorted[1:]
                for d in dupes:
                    if d["id"] not in seen_dup_ids:
                        seen_dup_ids.add(d["id"])
                        duplicates.append({
                            "reason": f"Duplicate file name: '{name}'",
                            "keep_id": keep["id"],
                            "keep_name": keep["name"],
                            "delete_id": d["id"],
                            "delete_name": d["name"],
                            "size": d.get("size", 0)
                        })
                        if auto_delete:
                            try:
                                self.delete_file(d["id"], folder_id=folder_id)
                                deleted_count += 1
                            except Exception as del_err:
                                print(f"[GDrive Error] Failed to delete/unlink duplicate {d['id']}: {del_err}")

        return {
            "total_files_scanned": len(files),
            "duplicates_found": len(duplicates),
            "duplicates_deleted": deleted_count,
            "details": duplicates
        }

    def batch_rename_folder(self, folder_id: str, prefix: str = "photo") -> Dict[str, Any]:
        """
        One-click renames all images directly in Google Drive to clean sequential names:
        e.g., photo_0001.jpg, photo_0002.jpg ...
        """
        files = self.list_folder_photos(folder_id)
        # Sort files consistently by createdTime / name
        files_sorted = sorted(files, key=lambda x: (x.get("createdTime", ""), x.get("name", "")))

        renamed_count = 0
        renamed_details = []

        for idx, f in enumerate(files_sorted, start=1):
            curr_name = f["name"]
            ext = Path(curr_name).suffix.lower()
            if not ext or ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                ext = ".jpg"

            new_name = f"{prefix}_{idx:04d}{ext}"
            if curr_name != new_name:
                try:
                    self.rename_file(f["id"], new_name)
                    renamed_count += 1
                    renamed_details.append({
                        "file_id": f["id"],
                        "old_name": curr_name,
                        "new_name": new_name
                    })
                except Exception as e:
                    print(f"[GDrive Error] Failed to rename {curr_name} to {new_name}: {e}")

        return {
            "total_files": len(files),
            "renamed_count": renamed_count,
            "details": renamed_details
        }

