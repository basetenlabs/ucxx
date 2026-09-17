# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import ucxx._lib.libucxx as ucx_api
from ucxx.exceptions import UCXError, UCXNoDeviceError

WORKER_DEVICE_KEYS = {
    "name",
    "transport",
    "index",
    "sys_device",
    "bandwidth",
    "latency",
    "overhead",
    "num_paths",
    "seg_size",
    "cap_flags",
}

ADDRESS_DEVICE_KEYS = {
    "index",
    "sys_device",
    "num_paths",
    "bandwidth",
    "latency",
    "overhead",
    "seg_size",
    "flags",
    "reachable_from_local",
    "device_address",
}


@pytest.fixture
def worker():
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    return ucx_api.UCXWorker(ctx)


def test_query_devices(worker):
    devices = worker.query_devices()

    assert devices
    for device in devices:
        assert set(device) == WORKER_DEVICE_KEYS
        assert device["name"]
        assert device["transport"]
        # reachable_from_local is a 64-bit bitmap keyed by this index, so an
        # index at or past 64 would be dropped from it with no error.
        assert device["index"] < 64


def test_query_devices_excludes_retired_device(worker):
    retired = worker.query_devices()[0]["name"]

    worker.exclude_device(retired)

    assert all(device["name"] != retired for device in worker.query_devices())


def test_query_address_devices(worker):
    entries = worker.query_address_devices(worker.address)

    assert entries
    known = 0
    for device in worker.query_devices():
        known |= 1 << device["index"]

    for entry in entries:
        assert set(entry) == ADDRESS_DEVICE_KEYS
        assert entry["index"] < 64
        assert isinstance(entry["device_address"], bytes)
        # Loopback settles no reachability. What is checkable here is that the
        # two queries agree: every bit set names a device query_devices() knows.
        assert entry["reachable_from_local"] & ~known == 0


def test_endpoint_transports(worker):
    ep = ucx_api.UCXEndpoint.create_from_worker_address(worker, worker.address, True)
    worker.progress()

    transports = ep.transports

    assert transports
    assert all(transport for transport, _device in transports)


def test_a_path_is_honoured(worker):
    address = worker.address
    entries = worker.query_address_devices(address)
    devices = worker.query_devices()

    # Every pair is tried, because most of them cannot carry this connection:
    # the listing's first device is `memory` on this node, an address advertises
    # `self/memory` too, and a path whose two ends cannot reach each other
    # affords no lane -- which UCX reports as unreachable rather than falling
    # back to a device the caller did not name. One working pair is the claim.
    #
    # Unskipped with the path API: the abort this used to hit -- several p2p
    # lanes drawing on one address entry -- is what selection now refuses.
    on_path = 0
    for device in devices:
        for entry in entries:
            try:
                ep = ucx_api.UCXEndpoint.create_from_worker_address_with_device(
                    worker,
                    address,
                    True,
                    None,
                    remote_device_index=entry["index"],
                    local_device_index=device["index"],
                )
            except UCXError:
                continue
            worker.progress()

            # Loopback decides no reachability, so the verdict reachable here is
            # that the path was accepted and lanes were still selected. Which
            # port carries bytes is the two-pod harness's answer.
            assert ep.transports
            on_path += 1

    assert on_path > 0


def test_a_path_rejects_an_unknown_peer_index(worker):
    address = worker.address
    unknown = max(entry["index"] for entry in worker.query_address_devices(address)) + 1
    devices = worker.query_devices()

    with pytest.raises(UCXNoDeviceError):
        ucx_api.UCXEndpoint.create_from_worker_address_with_device(
            worker,
            address,
            True,
            None,
            remote_device_index=unknown,
            local_device_index=devices[0]["index"],
        )


def test_half_a_path_names_this_end_only(worker):
    """A local device with no peer index is the one half a caller may name alone."""
    address = worker.address
    devices = worker.query_devices()

    named = 0
    for device in devices:
        try:
            ep = ucx_api.UCXEndpoint.create_from_worker_address_with_device(
                worker, address, True, device["name"]
            )
        except UCXError:
            continue
        worker.progress()
        assert ep.transports
        named += 1

    assert named > 0
