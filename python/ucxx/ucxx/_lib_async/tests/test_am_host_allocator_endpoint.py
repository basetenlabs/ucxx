# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import asyncio

import numpy as np
import pytest

import ucxx
from ucxx._lib_async.utils_test import wait_listener_client_handlers


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [64, 8 * 1024 * 1024])
async def test_an_allocator_registered_with_the_endpoint_is_given_the_receiving_one(size):
    calls = []

    def allocate(nbytes, ep_handle):
        calls.append((nbytes, ep_handle))
        return np.empty(nbytes, dtype=np.uint8)

    ucxx.core._get_ctx().worker.register_am_host_allocator(allocate, with_endpoint=True)
    received = asyncio.get_running_loop().create_future()

    async def server(ep):
        received.set_result(await ep.am_recv())
        await ep.close()

    listener = ucxx.create_listener(server)
    client = await ucxx.create_endpoint(ucxx.get_address(), listener.port)
    await client.am_send(np.full(size, 7, dtype=np.uint8))
    message = await asyncio.wait_for(received, 10)
    await wait_listener_client_handlers(listener)
    await client.close()

    np.testing.assert_equal(np.asarray(message), np.full(size, 7, dtype=np.uint8))
    ours = [ep_handle for nbytes, ep_handle in calls if nbytes == size]
    assert len(ours) == 1 and ours[0] != 0
