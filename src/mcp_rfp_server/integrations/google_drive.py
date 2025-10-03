import asyncio
import io
import logging
import os.path
from typing import Dict, List, Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger(__name__)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

class GoogleDriveClient:
    def __init__(self, credentials_path: str = "credentials.json", token_path: str = "token.json"):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = self._authenticate()
        logger.info("Google Drive client initialized successfully.")

    def _authenticate(self):
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Google API credentials not found at '{self.credentials_path}'.")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())
        try:
            return build("drive", "v3", credentials=creds)
        except HttpError as error:
            logger.error(f"An error occurred during Google Drive authentication: {error}")
            raise

    def list_files_in_folder(self, folder_id: str) -> List[Dict[str, Any]]:
        try:
            query = f"'{folder_id}' in parents and mimeType='text/plain' and trashed=false"
            results = self.service.files().list(q=query, fields="files(id, name)").execute()
            return results.get("files", [])
        except HttpError as error:
            logger.error(f"Failed to list files in Google Drive folder {folder_id}: {error}")
            return []

    def download_file_content(self, file_id: str) -> Optional[str]:
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return fh.getvalue().decode("utf-8")
        except HttpError as error:
            logger.error(f"Failed to download file {file_id}: {error}")
            return None

class GoogleDriveKnowledgeManager:
    def __init__(self, drive_client: GoogleDriveClient, folder_id: str):
        self.drive_client = drive_client
        self.folder_id = folder_id
        self._file_cache: Dict[str, Dict[str, Any]] = {}

    async def sync_knowledge_base(self) -> Dict[str, Any]:
        logger.info("Starting Google Drive knowledge base sync...")
        files = self.drive_client.list_files_in_folder(self.folder_id)
        synced_files, failed_files = [], []
        loop = asyncio.get_running_loop()
        for file_info in files:
            file_name = file_info["name"]
            try:
                content = await loop.run_in_executor(None, self.drive_client.download_file_content, file_info["id"])
                if content is not None:
                    self._file_cache[file_name] = {"content": content}
                    synced_files.append(file_name)
                else:
                    raise IOError("Downloaded content is None.")
            except Exception as e:
                failed_files.append({"name": file_name, "error": str(e)})
        return {"success": True, "synced_files": synced_files, "failed_files": failed_files}

    def get_file_content(self, filename: str) -> Optional[str]:
        return self._file_cache.get(filename, {}).get("content")