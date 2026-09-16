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


@pytest.mark.skip(
    reason="aborts inside UCX on A1's restricted path: wireup.c:412 "
    "`ep_addr_index < address->num_ep_addrs`, several p2p lanes landing on one "
    "address entry. Not reachable from anything in this layer."
)
def test_remote_device_pin_is_honoured(worker):
    address = worker.address
    entries = worker.query_address_devices(address)

    # Every advertised device is tried, because not all of them can carry this
    # connection: an address advertises `self/memory` too, and restricting an
    # endpoint to it leaves no usable transport, which UCX reports as
    # unreachable rather than falling back to another device.
    pinned = 0
    for entry in entries:
        try:
            ep = ucx_api.UCXEndpoint.create_from_worker_address_with_device(
                worker, address, True, None, entry["index"]
            )
        except UCXError:
            continue
        worker.progress()

        # Loopback decides no reachability, so the verdict reachable here is that
        # the pin was accepted and lanes were still selected. Which port carries
        # bytes is the two-pod harness's answer.
        assert ep.transports
        pinned += 1

    assert pinned > 0


def test_remote_device_pin_rejects_unknown_index(worker):
    address = worker.address
    unknown = max(entry["index"] for entry in worker.query_address_devices(address)) + 1

    with pytest.raises(UCXNoDeviceError):
        ucx_api.UCXEndpoint.create_from_worker_address_with_device(
            worker, address, True, None, unknown
        )
