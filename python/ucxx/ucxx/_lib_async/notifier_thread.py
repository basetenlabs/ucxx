# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause


import asyncio
import logging
import threading

import ucxx._lib.libucxx as ucx_api

logger = logging.getLogger("ucx")


async def _run_request_notifier(worker):
    return worker.run_request_notifier()


async def _notifier_coroutine(worker):
    worker.populate_python_futures_pool()
    finished = worker.wait_request_notifier()
    if finished:
        return True

    # Notify all enqueued waiting futures
    await _run_request_notifier(worker)

    return False


def _notifierThread(event_loop, worker, q):
    logger.debug("Starting Notifier Thread")
    asyncio.set_event_loop(event_loop)
    shutdown = False
    # Set once the loop has run a notification; bounds how long this thread waits
    # before looking for the next, as the 10 ms result wait did.
    notified = threading.Event()

    def notify():
        try:
            worker.run_request_notifier()
        finally:
            notified.set()

    while True:
        worker.populate_python_futures_pool()
        state = worker.wait_request_notifier(period_ns=int(1e9))  # 1 second timeout

        if not q.empty():
            q_val = q.get()
            if q_val == "shutdown":
                logger.debug("_notifierThread shutting down")
                shutdown = True
            else:
                logger.warning(
                    f"_notifierThread got unknown message from IPC queue: {q_val}"
                )

        if state == ucx_api.PythonRequestNotifierWaitState.Shutdown or shutdown is True:
            break
        elif state == ucx_api.PythonRequestNotifierWaitState.Timeout:
            continue

        # Notify all enqueued waiting futures, as one plain callback on the loop:
        # a coroutine would cost the loop a Task and a chained future per wakeup.
        notified.clear()
        try:
            event_loop.call_soon_threadsafe(notify)
        except RuntimeError as e:
            logger.debug(f"Notifier Thread could not reach the event loop: {e}")
            break
        if not notified.wait(0.01):
            # The callback stays scheduled: run_request_notifier() sets the asyncio
            # Futures of completed UCX requests, and dropping it would hang them.
            logger.debug("Notifier Thread notification still pending")

    # Clear all Python futures to ensure no references are held to the
    # `ucxx::Worker` that will prevent destructors from running.
    worker.clear_python_futures_pool()
