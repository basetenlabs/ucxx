/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <memory>
#include <vector>

#include <ucp/api/ucp.h>

#include <ucxx/remote_key.h>
#include <ucxx/request.h>
#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Put a group of spans, each under its own remote key, and complete once all are flushed.
 *
 * Every span is put with its memory type and memory handle named, so UCX never detects the
 * memory type. Submission runs on the worker's progress thread through delayed submission;
 * the request completes from the endpoint flush that follows the puts, so a Python future
 * is set once per group. The remote keys are held until that flush completes.
 */
class RequestMemPutv : public Request {
 private:
  std::vector<const void*> _buffers{};
  std::vector<size_t> _lengths{};
  std::vector<uint64_t> _remoteAddrs{};
  std::vector<std::shared_ptr<RemoteKey>> _remoteKeys{};
  ucs_memory_type_t _memoryType{UCS_MEMORY_TYPE_UNKNOWN};
  std::vector<ucp_mem_h> _memoryHandles{};
  ucs_status_t _submitStatus{UCS_OK};  ///< First put that failed to submit; the flush still runs

  RequestMemPutv(std::shared_ptr<Endpoint> endpoint,
                 std::vector<const void*> buffers,
                 std::vector<size_t> lengths,
                 std::vector<uint64_t> remoteAddrs,
                 std::vector<std::shared_ptr<RemoteKey>> remoteKeys,
                 ucs_memory_type_t memoryType,
                 std::vector<ucp_mem_h> memoryHandles,
                 const bool enablePythonFuture,
                 RequestCallbackUserFunction callbackFunction,
                 RequestCallbackUserData callbackData);

  static void flushCallback(void* request, ucs_status_t status, void* arg);

  void request();

 public:
  /**
   * @brief Create a request putting every span `i` from `buffers[i]` to `remoteAddrs[i]` under
   * `remoteKeys[i]`.
   *
   * @throws std::invalid_argument if the vectors differ in length, are empty, or a memory
   *         handle or remote key is null.
   */
  friend std::shared_ptr<RequestMemPutv> createRequestMemPutv(
    std::shared_ptr<Endpoint> endpoint,
    std::vector<const void*> buffers,
    std::vector<size_t> lengths,
    std::vector<uint64_t> remoteAddrs,
    std::vector<std::shared_ptr<RemoteKey>> remoteKeys,
    ucs_memory_type_t memoryType,
    std::vector<ucp_mem_h> memoryHandles,
    const bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);

  void populateDelayedSubmission() override;
};

[[nodiscard]] std::shared_ptr<RequestMemPutv> createRequestMemPutv(
  std::shared_ptr<Endpoint> endpoint,
  std::vector<const void*> buffers,
  std::vector<size_t> lengths,
  std::vector<uint64_t> remoteAddrs,
  std::vector<std::shared_ptr<RemoteKey>> remoteKeys,
  ucs_memory_type_t memoryType,
  std::vector<ucp_mem_h> memoryHandles,
  const bool enablePythonFuture                = false,
  RequestCallbackUserFunction callbackFunction = nullptr,
  RequestCallbackUserData callbackData         = nullptr);

}  // namespace ucxx
