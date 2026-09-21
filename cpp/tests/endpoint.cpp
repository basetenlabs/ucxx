/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>

namespace {

class EndpointTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Context> _remoteContext{
    ucxx::createContext({}, ucxx::Context::defaultFeatureFlags)};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Worker> _remoteWorker{nullptr};

  virtual void SetUp()
  {
    _worker       = _context->createWorker();
    _remoteWorker = _remoteContext->createWorker();
  }
};

TEST_F(EndpointTest, HandleIsValid)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  _worker->progress();

  ASSERT_TRUE(ep->getHandle() != nullptr);
}

TEST_F(EndpointTest, IsAlive)
{
  GTEST_SKIP()
    << "Connecting to worker via its UCX address doesn't seem to call endpoint error handler";
  auto ep = _worker->createEndpointFromWorkerAddress(_remoteWorker->getAddress());
  _worker->progress();
  _remoteWorker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> buf{123};
  auto send_req = ep->tagSend(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0});
  while (!send_req->isCompleted())
    _worker->progress();

  _remoteWorker  = nullptr;
  _remoteContext = nullptr;
  _worker->progress();
  ASSERT_FALSE(ep->isAlive());
}

TEST_F(EndpointTest, GetTransports)
{
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  _worker->progress();

  auto transports = ep->getTransports();

  ASSERT_FALSE(transports.empty());
  for (const auto& transport : transports)
    EXPECT_FALSE(transport.transport.empty());
}

TEST_F(EndpointTest, QueryTransportsByHandle)
{
  // The reading b10 needs for an endpoint it holds only as a `ucp_ep_h`: the
  // one a peer's wireup built, which ucxx never created. Nothing here can
  // produce such an endpoint, so what is checked is that the handle route
  // answers what the owning endpoint answers.
  auto ep = _worker->createEndpointFromWorkerAddress(_worker->getAddress());
  _worker->progress();

  auto byHandle =
    _worker->queryEndpointTransports(reinterpret_cast<uintptr_t>(ep->getHandle()));
  auto byEndpoint = ep->getTransports();

  ASSERT_EQ(byEndpoint.size(), byHandle.size());
  for (size_t i = 0; i < byHandle.size(); ++i) {
    EXPECT_EQ(byEndpoint[i].transport, byHandle[i].transport);
    EXPECT_EQ(byEndpoint[i].device, byHandle[i].device);
  }
}

TEST_F(EndpointTest, QueryTransportsRejectsNullHandle)
{
  EXPECT_THROW(_worker->queryEndpointTransports(0), ucxx::Error);
}

// Disabled, not deleted: this aborts the whole binary inside UCX, on A1's
// restricted path and not in any code here --
//   wireup.c:412  Assertion `ep_addr_index < address->num_ep_addrs' failed:
//   lane=2/5 tl_name_csum=0xd47a address_index=0 ep_addr_index=1 num_ep_addrs=1
// Confining lane selection to one remote device puts several p2p lanes on one
TEST_F(EndpointTest, PathIsHonoured)
{
  auto address = _remoteWorker->getAddress();
  auto entries = _worker->queryAddressDevices(address);
  auto devices = _worker->queryDevices();
  ASSERT_FALSE(entries.empty());
  ASSERT_FALSE(devices.empty());

  // Every advertised device is tried, because not all of them can carry this
  // connection: an address advertises `self/memory` too, and restricting an
  // endpoint between two workers to it leaves no usable transport, which UCX
  // reports as unreachable rather than falling back to another device.
  size_t on_path = 0;
  for (const auto& entry : entries) {
    std::shared_ptr<ucxx::Endpoint> ep;
    try {
      ep = _worker->createEndpointFromWorkerAddressOnPath(
        address, true, devices[0].index, entry.index);
    } catch (const ucxx::Error&) {
      continue;
    }
    _worker->progress();
    _remoteWorker->progress();

    // Loopback decides no reachability, so the verdict this test can reach is
    // that the pin was accepted and lanes were still selected; that the bytes
    // leave by the named port is the two-pod harness's to answer.
    EXPECT_FALSE(ep->getTransports().empty());
    ++on_path;
  }
  EXPECT_GT(on_path, 0u);
}

TEST_F(EndpointTest, PathRejectsUnknownPeerIndex)
{
  auto address = _remoteWorker->getAddress();
  auto entries = _worker->queryAddressDevices(address);
  auto devices = _worker->queryDevices();
  ASSERT_FALSE(entries.empty());
  ASSERT_FALSE(devices.empty());

  unsigned unknown = 0;
  for (const auto& entry : entries)
    unknown = std::max(unknown, entry.index + 1);

  EXPECT_THROW(std::ignore = _worker->createEndpointFromWorkerAddressOnPath(
                 address, true, devices[0].index, unknown),
               ucxx::Error);
}

TEST(AddressTest, EmptyAddressRejected)
{
  EXPECT_THROW(std::ignore = ucxx::createAddressFromString(""), std::invalid_argument);
}

}  // namespace
