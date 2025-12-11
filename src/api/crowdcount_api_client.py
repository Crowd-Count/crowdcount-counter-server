"""
CrowdCount API Client
Handles API calls to the CrowdCount backend server
"""

import requests
import logging
from typing import Optional


class CrowdCountAPIClient:
    """
    Client for making API calls to CrowdCount backend
    Each device/stream has its own credentials
    """

    def __init__(self, device_id: str, password: str, base_url: str = 'https://api.crowdcount.tech', timeout: int = 5):
        """
        Initialize API client with device credentials

        Args:
            device_id: Unique device identifier
            password: Device password for authentication
            base_url: Base URL for the CrowdCount API
            timeout: Request timeout in seconds
        """
        self.device_id = device_id
        self.password = password
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.enabled = bool(device_id and password)  # Only enabled if both credentials provided

        # Setup logging (only add handler and log once per device_id)
        self.logger = logging.getLogger(f"APIClient-{device_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - {device_id} - API - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            # Only log initialization once per device
            if self.enabled:
                self.logger.info(f"API Client initialized for device: {device_id}")
            else:
                self.logger.info(f"API Client disabled (no credentials provided)")

    def add_to_count(self, count: int = 1) -> bool:
        """
        Add to the count (person entered)

        Args:
            count: Number to add to count (default: 1)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return True  # Silently succeed if API is disabled

        try:
            url = f"{self.base_url}/add_to_count"
            data = {
                'user_id': self.device_id,
                'password': self.password,
                'count_to_add': str(count)
            }

            self.logger.debug(f"POST {url} - Adding {count} to count")
            response = requests.post(url, data=data, timeout=self.timeout)

            if response.status_code == 200:
                self.logger.info(f"✓ Added {count} to remote count")
                return True
            else:
                self.logger.warning(f"✗ API Error: HTTP {response.status_code}")
                return False

        except requests.Timeout:
            self.logger.warning(f"✗ API Timeout: Request took longer than {self.timeout}s")
            return False
        except requests.ConnectionError:
            self.logger.warning(f"✗ API Connection Error: Could not reach {self.base_url}")
            return False
        except Exception as e:
            self.logger.warning(f"✗ API Error: {e}")
            return False

    def subtract_from_count(self, count: int = 1) -> bool:
        """
        Subtract from the count (person exited)

        Args:
            count: Number to subtract from count (default: 1)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return True  # Silently succeed if API is disabled

        try:
            url = f"{self.base_url}/sub_from_count"
            data = {
                'user_id': self.device_id,
                'password': self.password,
                'count_to_sub': str(count)
            }

            self.logger.debug(f"POST {url} - Subtracting {count} from count")
            response = requests.post(url, data=data, timeout=self.timeout)

            if response.status_code == 200:
                self.logger.info(f"✓ Subtracted {count} from remote count")
                return True
            else:
                self.logger.warning(f"✗ API Error: HTTP {response.status_code}")
                return False

        except requests.Timeout:
            self.logger.warning(f"✗ API Timeout: Request took longer than {self.timeout}s")
            return False
        except requests.ConnectionError:
            self.logger.warning(f"✗ API Connection Error: Could not reach {self.base_url}")
            return False
        except Exception as e:
            self.logger.warning(f"✗ API Error: {e}")
            return False

    def get_current_count(self) -> Optional[int]:
        """
        Get the current count from the API

        Returns:
            Current count as integer, or None if request failed
        """
        if not self.enabled:
            return None

        try:
            url = f"{self.base_url}/get_count"
            params = {
                'user_id': self.device_id,
                'password': self.password
            }

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                count = data.get('count')
                self.logger.debug(f"Current count: {count}")
                return count
            else:
                self.logger.warning(f"✗ API Error: HTTP {response.status_code}")
                return None

        except Exception as e:
            self.logger.warning(f"✗ API Error: {e}")
            return None

    def set_enabled(self, enabled: bool):
        """
        Enable or disable API calls

        Args:
            enabled: True to enable, False to disable
        """
        self.enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"API calls {status}")

    def update_status(self, status: str) -> bool:
        """
        Update device online/offline status on the server

        Args:
            status: Either "online" or "offline"

        Returns:
            True if successful, False otherwise
        """
        if status not in {"online", "offline"}:
            self.logger.error(f"Invalid status: {status}. Must be 'online' or 'offline'")
            return False

        # Always try to update status, even if API is "disabled" for counting
        # Status updates are important for device management
        if not self.device_id or not self.password:
            self.logger.warning("Cannot update status: no credentials")
            return False

        try:
            url = f"{self.base_url}/update_status"
            data = {
                'user_id': self.device_id,
                'password': self.password,
                'status': status
            }

            self.logger.debug(f"POST {url} - Updating status to {status}")
            response = requests.post(url, data=data, timeout=self.timeout)

            if response.status_code == 200:
                self.logger.info(f"✓ Device status updated to {status}")
                return True
            else:
                self.logger.warning(f"✗ Status Update Error: HTTP {response.status_code}")
                return False

        except requests.Timeout:
            self.logger.warning(f"✗ Status Update Timeout: Request took longer than {self.timeout}s")
            return False
        except requests.ConnectionError:
            self.logger.warning(f"✗ Status Update Connection Error: Could not reach {self.base_url}")
            return False
        except Exception as e:
            self.logger.warning(f"✗ Status Update Error: {e}")
            return False

    def get_direction_flipped(self) -> bool:
        """
        Get the direction_flipped setting from the server

        Returns:
            True if direction is flipped, False otherwise (defaults to False on error)
        """
        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/get_direction_flipped"
            params = {
                'user_id': self.device_id,
                'password': self.password
            }

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                flipped = data.get('direction_flipped', False)
                self.logger.debug(f"Direction flipped setting: {flipped}")
                return bool(flipped)
            else:
                self.logger.warning(f"✗ Failed to get direction_flipped: HTTP {response.status_code}")
                return False

        except Exception as e:
            self.logger.warning(f"✗ Error getting direction_flipped: {e}")
            return False
