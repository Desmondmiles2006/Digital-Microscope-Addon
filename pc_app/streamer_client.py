"""
streamer_client.py — ESP32-CAM JPEG polling client

Polls /stream endpoint repeatedly to simulate live video.
Auto-reconnects with exponential backoff on connection loss.
"""

import time
import threading
import logging
from typing import Callable, Optional

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


class MJPEGClient:
    """
    Polls the ESP32-CAM /stream endpoint repeatedly and delivers
    decoded OpenCV BGR frames to a registered callback.

    Args:
        host: ESP32-CAM IP address (e.g. 192.168.1.4)
        port: HTTP port (default 80)
        on_frame: callback(frame: np.ndarray) called for each decoded frame
    """

    def __init__(
        self,
        host: str,
        port: int = 80,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.on_frame = on_frame
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps = 0.0
        self._frame_times: list[float] = []

    @property
    def stream_url(self) -> str:
        return "http://" + self.host + ":" + str(self.port) + "/stream"

    @property
    def capture_url(self) -> str:
        return "http://" + self.host + ":" + str(self.port) + "/capture"

    @property
    def status_url(self) -> str:
        return "http://" + self.host + ":" + str(self.port) + "/status"

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    def start(self) -> None:
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("MJPEGClient started: %s", self.stream_url)

    def stop(self) -> None:
        """Stop polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def capture_single(self) -> Optional[np.ndarray]:
        """Fetch a single JPEG from the /capture endpoint."""
        try:
            resp = requests.get(self.capture_url, timeout=5)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.error("capture_single failed: %s", exc)
            return None

    def get_status(self) -> dict:
        """Fetch JSON status from the /status endpoint."""
        try:
            resp = requests.get(self.status_url, timeout=3)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("get_status failed: %s", exc)
            return {}

    def _update_fps(self) -> None:
        now = time.time()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        self._fps = float(len(self._frame_times))

    def _poll_loop(self) -> None:
        """Poll /stream endpoint in a loop to get live frames."""
        backoff = 1.0
        session = requests.Session()

        while self._running:
            try:
                resp = session.get(self.stream_url, timeout=5)
                resp.raise_for_status()

                arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    self._update_fps()
                    backoff = 1.0
                    if self.on_frame:
                        self.on_frame(frame)
                else:
                    logger.warning("Frame decode failed")
                    time.sleep(0.1)

            except Exception as exc:
                if self._running:
                    logger.warning(
                        "Poll error: %s. Retrying in %.1f seconds.", exc, backoff
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5.0)