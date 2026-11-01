# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import os

import pytest

from ucxx._lib import libucxx as ucx_api
from ucxx.testing import wait_requests

SPAN = 4096


def _address(buffer: bytearray) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


@pytest.fixture
def loopback():
    """A worker with an endpoint to itself, and a source and destination buffer, each mapped."""
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.RMA, ucx_api.Feature.WAKEUP))
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, worker.address, endpoint_error_handling=True
    )
    src = bytearray(os.urandom(3 * SPAN))
    dst = bytearray(3 * SPAN)
    host = ucx_api.MemoryType.HOST
    src_handle = ucx_api.UCXMemoryHandle.create(ctx, _address(src), len(src), host)
    dst_handle = ucx_api.UCXMemoryHandle.create(ctx, _address(dst), len(dst), host)
    remote_key = ucx_api.UCXRemoteKey.from_serialized(ep, dst_handle.remote_key().serialize())
    yield worker, ep, src, dst, src_handle, remote_key, ctx
    del remote_key, src_handle, dst_handle
    ep.close_blocking(period=10**10, max_attempts=1)


def test_every_span_lands_at_its_own_remote_address(loopback):
    worker, ep, src, dst, src_handle, remote_key, ctx = loopback
    # Spans land out of order, so a put that ignored its remote address would show.
    order = [2, 0, 1]

    request = ep.mem_putv(
        [_address(src) + i * SPAN for i in range(3)],
        [SPAN] * 3,
        [_address(dst) + order[i] * SPAN for i in range(3)],
        remote_key,
        ucx_api.MemoryType.HOST,
        [src_handle.handle] * 3,
    )
    wait_requests(worker, "blocking", request)

    for i in range(3):
        assert dst[order[i] * SPAN : (order[i] + 1) * SPAN] == src[i * SPAN : (i + 1) * SPAN]


def test_spans_under_different_remote_keys_land_in_their_own_regions(loopback):
    """Each region has its own key, as each layer of a KV pool does; one request reaches both."""
    worker, ep, src, dst, src_handle, remote_key, ctx = loopback
    other = bytearray(SPAN)
    other_handle = ucx_api.UCXMemoryHandle.create(
        ctx, _address(other), len(other), ucx_api.MemoryType.HOST
    )
    other_key = ucx_api.UCXRemoteKey.from_serialized(ep, other_handle.remote_key().serialize())

    request = ep.mem_putv(
        [_address(src), _address(src) + SPAN],
        [SPAN, SPAN],
        [_address(dst), _address(other)],
        [remote_key, other_key],
        ucx_api.MemoryType.HOST,
        [src_handle.handle] * 2,
    )
    wait_requests(worker, "blocking", request)

    assert dst[:SPAN] == src[:SPAN] and other == src[SPAN : 2 * SPAN]
    del other_key, other_handle


def test_a_span_without_its_memory_handle_is_refused(loopback):
    worker, ep, src, dst, src_handle, remote_key, ctx = loopback

    with pytest.raises(ValueError, match="memory handle"):
        ep.mem_putv(
            [_address(src), _address(src) + SPAN],
            [SPAN, SPAN],
            [_address(dst), _address(dst) + SPAN],
            remote_key,
            ucx_api.MemoryType.HOST,
            [src_handle.handle],
        )
