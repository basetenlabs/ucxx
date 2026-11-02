/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/endpoint.h>
#include <ucxx/log.h>
#include <ucxx/request_mem_putv.h>
#include <ucxx/worker.h>

namespace ucxx {

std::shared_ptr<RequestMemPutv> createRequestMemPutv(
  std::shared_ptr<Endpoint> endpoint,
  std::vector<const void*> buffers,
  std::vector<size_t> lengths,
  std::vector<uint64_t> remoteAddrs,
  std::vector<std::shared_ptr<RemoteKey>> remoteKeys,
  ucs_memory_type_t memoryType,
  std::vector<ucp_mem_h> memoryHandles,
  const bool enablePythonFuture,
  RequestCallbackUserFunction callbackFunction,
  RequestCallbackUserData callbackData)
{
  auto req = std::shared_ptr<RequestMemPutv>(new RequestMemPutv(endpoint,
                                                                std::move(buffers),
                                                                std::move(lengths),
                                                                std::move(remoteAddrs),
                                                                std::move(remoteKeys),
                                                                memoryType,
                                                                std::move(memoryHandles),
                                                                enablePythonFuture,
                                                                callbackFunction,
                                                                callbackData));

  // Submitted by the progress thread, which also sets the Python future, so nothing here
  // takes the GIL.
  req->_worker->registerDelayedSubmission(
    req, std::bind(std::mem_fn(&Request::populateDelayedSubmission), req.get()));

  return req;
}

RequestMemPutv::RequestMemPutv(std::shared_ptr<Endpoint> endpoint,
                               std::vector<const void*> buffers,
                               std::vector<size_t> lengths,
                               std::vector<uint64_t> remoteAddrs,
                               std::vector<std::shared_ptr<RemoteKey>> remoteKeys,
                               ucs_memory_type_t memoryType,
                               std::vector<ucp_mem_h> memoryHandles,
                               const bool enablePythonFuture,
                               RequestCallbackUserFunction callbackFunction,
                               RequestCallbackUserData callbackData)
  : Request(
      endpoint, data::RequestData{}, "memPutv", enablePythonFuture, callbackFunction, callbackData),
    _buffers(std::move(buffers)),
    _lengths(std::move(lengths)),
    _remoteAddrs(std::move(remoteAddrs)),
    _remoteKeys(std::move(remoteKeys)),
    _memoryType(memoryType),
    _memoryHandles(std::move(memoryHandles))
{
  if (_endpoint == nullptr)
    throw std::invalid_argument("A valid endpoint is required to put memory.");
  const size_t spans = _buffers.size();
  if (spans == 0 || _lengths.size() != spans || _remoteAddrs.size() != spans ||
      _remoteKeys.size() != spans || _memoryHandles.size() != spans)
    throw std::invalid_argument(
      "memPutv needs one length, remote address, remote key and memory handle per span, and one "
      "span at least.");
  for (const auto memoryHandle : _memoryHandles)
    if (memoryHandle == nullptr)
      throw std::invalid_argument("memPutv needs a memory handle for every span.");
  for (const auto& remoteKey : _remoteKeys)
    if (remoteKey == nullptr)
      throw std::invalid_argument("memPutv needs a remote key for every span.");
}

void RequestMemPutv::flushCallback(void* request, ucs_status_t status, void* arg)
{
  auto req = reinterpret_cast<RequestMemPutv*>(arg);
  req->callback(request, req->_submitStatus != UCS_OK ? req->_submitStatus : status);
}

void RequestMemPutv::request()
{
  auto endpointHandle = _endpoint->getHandle();
  for (size_t i = 0; i < _buffers.size(); ++i) {
    ucp_request_param_t param = {
      .op_attr_mask = UCP_OP_ATTR_FIELD_MEMORY_TYPE | UCP_OP_ATTR_FIELD_MEMH,
      .memory_type  = _memoryType,
      .memh         = _memoryHandles[i],
    };
    void* put = ucp_put_nbx(endpointHandle,
                            _buffers[i],
                            _lengths[i],
                            _remoteAddrs[i],
                            _remoteKeys[i]->getHandle(),
                            &param);
    if (UCS_PTR_IS_ERR(put)) {
      _submitStatus = UCS_PTR_STATUS(put);
      break;
    }
    // In flight: the flush below completes it, so its own handle is not needed.
    if (put != nullptr) ucp_request_free(put);
  }

  // Flushed even when a put failed to submit: the ones already submitted read their remote keys,
  // which this request holds until the flush completes.
  ucp_request_param_t param = {
    .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA, .user_data = this};
  param.cb.send = flushCallback;
  void* flush   = ucp_ep_flush_nbx(endpointHandle, &param);
  if (!UCS_PTR_IS_PTR(flush) && !UCS_PTR_IS_ERR(flush) && _submitStatus != UCS_OK)
    flush = UCS_STATUS_PTR(_submitStatus);
  publishRequest(flush);
}

void RequestMemPutv::populateDelayedSubmission()
{
  if (_endpoint->getHandle() == nullptr) {
    ucxx_warn("Endpoint was closed before memory could be put");
    Request::callback(this, UCS_ERR_CANCELED);
    return;
  }

  request();
  process();
}

}  // namespace ucxx
