"""
SharePoint Integration for MCP-RFP Server with SSL Fix
"""
import asyncio
import logging
import ssl
from typing import Dict, List, Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class SharePointClient:
    """Client for Microsoft SharePoint integration via Graph API"""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str, site_url: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.site_url = site_url
        self.access_token = None
        self.session = None
        self._initialized = False

    async def initialize(self):
        """Initialize SharePoint client and authenticate"""
        if self._initialized:
            return

        try:
            # Create SSL context that skips verification for testing
            # WARNING: This is not secure for production use
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self.session = aiohttp.ClientSession(connector=connector)
            await self._authenticate()
            self._initialized = True
            logger.info("SharePoint client initialized successfully")
        except Exception as e:
            logger.error(f"SharePoint client initialization failed: {e}")
            raise

    async def _authenticate(self):
        """Authenticate with Microsoft Graph API using client credentials"""
        auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'https://graph.microsoft.com/.default'
        }

        async with self.session.post(auth_url, data=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Authentication failed: {response.status} - {error_text}")

            auth_data = await response.json()
            self.access_token = auth_data['access_token']
            logger.info("Successfully authenticated with SharePoint")

    async def get_site_id(self) -> str:
        """Get the SharePoint site ID from the site URL"""
        await self.initialize()

        # Extract hostname and site path from URL
        # Expected format: https://yourdomain.sharepoint.com/sites/sitename
        if '/sites/' in self.site_url:
            hostname = self.site_url.split('/sites/')[0].replace('https://', '')
            site_path = '/sites/' + self.site_url.split('/sites/')[1]
        else:
            # Root site
            hostname = self.site_url.replace('https://', '')
            site_path = '/'

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

        # Use Graph API to get site info
        graph_url = f"https://graph.microsoft.com/v1.0/sites/{hostname}:{site_path}"

        async with self.session.get(graph_url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to get site ID: {response.status} - {error_text}")

            site_data = await response.json()
            return site_data['id']

    async def list_files_in_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """List all files in a SharePoint folder"""
        await self.initialize()

        try:
            site_id = await self.get_site_id()

            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }

            # Get the drive (document library)
            drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"

            async with self.session.get(drives_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get drives: {response.status} - {error_text}")

                drives_data = await response.json()

                # Find the Documents drive (default document library)
                documents_drive = None
                for drive in drives_data['value']:
                    if drive['name'] == 'Documents' or drive['webUrl'].endswith('/Shared Documents'):
                        documents_drive = drive
                        break

                if not documents_drive:
                    raise Exception("Documents library not found")

            # List files in the specified folder
            folder_url = f"https://graph.microsoft.com/v1.0/drives/{documents_drive['id']}/root:/{folder_path}:/children"

            async with self.session.get(folder_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to list folder contents: {response.status} - {error_text}")

                folder_data = await response.json()
                return folder_data.get('value', [])

        except Exception as e:
            logger.error(f"Failed to list files in folder {folder_path}: {e}")
            raise

    async def download_file_content(self, file_id: str) -> str:
        """Download file content from SharePoint"""
        await self.initialize()

        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

        # Get file content
        content_url = f"https://graph.microsoft.com/v1.0/drives/{file_id.split(':')[0]}/items/{file_id}/content"

        async with self.session.get(content_url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to download file: {response.status} - {error_text}")

            return await response.text()

    async def get_knowledge_base_files(self, folder_path: str = "knowledge_base") -> List[Dict[str, Any]]:
        """Get all .txt files from the knowledge base folder in SharePoint"""
        await self.initialize()

        try:
            files = await self.list_files_in_folder(folder_path)

            # Filter for .txt files only
            txt_files = []
            for file in files:
                if file.get('file') and file['name'].lower().endswith('.txt'):
                    txt_files.append({
                        'id': file['id'],
                        'name': file['name'],
                        'size': file['size'],
                        'download_url': file.get('@microsoft.graph.downloadUrl'),
                        'web_url': file.get('webUrl'),
                        'last_modified': file.get('lastModifiedDateTime')
                    })

            logger.info(f"Found {len(txt_files)} .txt files in SharePoint folder: {folder_path}")
            return txt_files

        except Exception as e:
            logger.error(f"Failed to get knowledge base files: {e}")
            raise

    async def download_file_by_url(self, download_url: str) -> str:
        """Download file content using the direct download URL"""
        if not download_url:
            raise Exception("No download URL provided")

        async with self.session.get(download_url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Failed to download file from URL: {response.status} - {error_text}")

            return await response.text(encoding='utf-8')

    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class SharePointKnowledgeManager:
    """Manager for SharePoint-based knowledge base operations"""

    def __init__(self, sharepoint_client: SharePointClient, folder_path: str = "knowledge_base"):
        self.sharepoint_client = sharepoint_client
        self.folder_path = folder_path
        self._file_cache = {}
        self._last_sync = None

    async def sync_knowledge_base(self) -> Dict[str, Any]:
        """Sync knowledge base files from SharePoint"""
        try:
            logger.info("Starting SharePoint knowledge base sync...")

            # Get all .txt files from SharePoint
            files = await self.sharepoint_client.get_knowledge_base_files(self.folder_path)

            synced_files = []
            failed_files = []

            for file_info in files:
                try:
                    logger.info(f"Downloading: {file_info['name']}")

                    # Download file content
                    if file_info.get('download_url'):
                        content = await self.sharepoint_client.download_file_by_url(
                            file_info['download_url']
                        )
                    else:
                        # Fallback method if download URL not available
                        content = await self.sharepoint_client.download_file_content(file_info['id'])

                    # Cache the content
                    self._file_cache[file_info['name']] = {
                        'content': content,
                        'size': file_info['size'],
                        'last_modified': file_info['last_modified'],
                        'web_url': file_info.get('web_url')
                    }

                    synced_files.append(file_info['name'])

                except Exception as e:
                    logger.error(f"Failed to download {file_info['name']}: {e}")
                    failed_files.append({'name': file_info['name'], 'error': str(e)})

            self._last_sync = asyncio.get_event_loop().time()

            logger.info(f"SharePoint sync completed: {len(synced_files)} files synced, {len(failed_files)} failed")

            return {
                'success': True,
                'synced_files': synced_files,
                'failed_files': failed_files,
                'total_files': len(files),
                'sync_time': self._last_sync
            }

        except Exception as e:
            logger.error(f"SharePoint knowledge base sync failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'synced_files': [],
                'failed_files': []
            }

    def get_file_content(self, filename: str) -> Optional[str]:
        """Get cached file content"""
        file_data = self._file_cache.get(filename)
        return file_data['content'] if file_data else None

    def list_cached_files(self) -> List[str]:
        """Get list of cached filenames"""
        return list(self._file_cache.keys())

    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get cached file information"""
        return self._file_cache.get(filename)

    def is_synced(self) -> bool:
        """Check if knowledge base has been synced"""
        return self._last_sync is not None and len(self._file_cache) > 0