# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from typing import Callable
from vllm.logger import init_logger

logger = init_logger(__name__)


class AsyncMaxConfidenceThresholdUpdater:
    def __init__(
        self,
        update_fn: Callable[[], None],
        debounce_seconds: float = 1,
    ) -> None:
        self._update_fn = update_fn
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._shutdown = False
        self._pending = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="confidence-threshold-updater",
            daemon=True,
        )
        self._worker.start()
        logger.debug(
            "Started async max-confidence updater worker (debounce_seconds=%s).",
            self._debounce_seconds,
        )

    def mark_should_update(self) -> None:
        with self._condition:
            if self._shutdown:
                logger.debug(
                    "Ignoring confidence-threshold update mark because worker is shutting down."
                )
                return
            was_pending = self._pending
            self._pending = True
            self._condition.notify()
        if was_pending:
            logger.debug("Coalescing confidence-threshold update mark into existing pending update.")
        else:
            logger.debug("Marked confidence-threshold update and notified background worker.")

    def shutdown(self) -> None:
        logger.debug("Shutting down async max-confidence updater worker.")
        with self._condition:
            self._shutdown = True
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
            if self._worker.is_alive():
                logger.warning("Async max-confidence updater worker did not exit within timeout.")
            else:
                logger.debug("Async max-confidence updater worker exited cleanly.")

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._shutdown and not self._pending:
                    self._condition.wait()

                if self._shutdown:
                    logger.debug("Async max-confidence updater worker received shutdown signal.")
                    return

                # Consume the current pending mark.
                self._pending = False

                logger.debug(
                    "Debouncing confidence-threshold updates for up to %s seconds.",
                    self._debounce_seconds,
                )
                self._condition.wait(timeout=self._debounce_seconds)

                if self._shutdown:
                    logger.debug("Async max-confidence updater worker stopping during debounce wait.")
                    return

                # Consume any marks that arrived during debounce.
                self._pending = False

            try:
                logger.debug("Running confidence-threshold update in background.")
                self._update_fn()
                time.sleep(self._debounce_seconds)
            except Exception:
                logger.exception("Confidence-threshold update failed in background worker.")