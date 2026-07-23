/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <utility>
#include <vector>

#include <ucxx/inflight_requests.h>
#include <ucxx/log.h>
#include <ucxx/request.h>

namespace ucxx {

InflightRequests::~InflightRequests() { cancelAll(); }

size_t InflightRequests::size()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _inflight.size();
}

void InflightRequests::insert(const std::shared_ptr<Request>& request)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _inflight.insert(request);
}

size_t InflightRequests::cancelingSize()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _canceling.size();
}

void InflightRequests::remove(const std::shared_ptr<Request>& request)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _inflight.erase(request);
  _canceling.erase(request);
}

void InflightRequests::merge(TrackedRequests&& trackedRequests)
{
  std::lock_guard<std::mutex> lock(_mutex);
  for (auto& r : trackedRequests.inflight)
    if (r) _inflight.insert(std::move(r));
  for (auto& r : trackedRequests.canceling)
    if (r) _canceling.insert(std::move(r));
}

size_t InflightRequests::cancelAll()
{
  decltype(_inflight) toCancel;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    toCancel = std::exchange(_inflight, {});
  }

  size_t total = toCancel.size();
  if (total == 0) return 0;

  ucxx_debug("ucxx::InflightRequests::%s, canceling %lu requests", __func__, total);

  for (auto& r : toCancel) {
    if (r) r->cancel();
  }

  {
    std::lock_guard<std::mutex> lock(_mutex);
    for (auto& r : toCancel) {
      if (r && r->getStatus() == UCS_INPROGRESS) _canceling.insert(r);
    }
  }

  return total;
}

TrackedRequests InflightRequests::release()
{
  std::lock_guard<std::mutex> lock(_mutex);
  TrackedRequests result;

  result.inflight.reserve(_inflight.size());
  for (auto& r : _inflight)
    result.inflight.push_back(r);
  _inflight.clear();

  result.canceling.reserve(_canceling.size());
  for (auto& r : _canceling)
    result.canceling.push_back(r);
  _canceling.clear();

  return result;
}

size_t InflightRequests::getCancelingSize()
{
  /* Two-phase: snapshot under the container lock, query request status
   * without it, erase completed entries under the lock again.
   *
   * The phases exist to avoid an AB-BA deadlock with the completion path:
   *
   * - Thread A (request completes): `Request::setStatus()` holds the
   *   REQUEST mutex, then calls `Worker::removeInflightRequest()` ->
   *   `InflightRequests::remove()`, which takes this CONTAINER mutex.
   *   Order: request -> container.
   * - Thread B (this function, single-lock version): holds the CONTAINER
   *   mutex while `r->getStatus()` takes the REQUEST mutex.
   *   Order: container -> request.
   *
   * If A completes the exact request B is querying, A waits for the
   * container mutex while B waits for that request's mutex - both stuck
   * forever, wedging the progress thread (which runs the completion
   * callbacks). Parked requests complete in bursts during endpoint
   * teardown, which is also when new batches are parked and pruned, so
   * the single-lock version is not a rare race.
   *
   * TODO: clean up this two-phase workaround by making
   * `Request::_status` a `std::atomic` so `getStatus()` is a lock-free
   * read; then this function can hold the container lock for a single
   * simple pass (and the same inversion between
   * `Endpoint::cancelInflightRequestsBlocking` and completing requests
   * on the endpoint's container disappears class-wide). */
  std::vector<std::shared_ptr<Request>> snapshot;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    snapshot.reserve(_canceling.size());
    for (auto& r : _canceling)
      snapshot.push_back(r);
  }

  std::vector<std::shared_ptr<Request>> completed;
  for (auto& r : snapshot)
    if (!r || r->getStatus() != UCS_INPROGRESS) completed.push_back(r);

  std::lock_guard<std::mutex> lock(_mutex);
  for (auto& r : completed)
    _canceling.erase(r);

  return _canceling.size();
}

}  // namespace ucxx
