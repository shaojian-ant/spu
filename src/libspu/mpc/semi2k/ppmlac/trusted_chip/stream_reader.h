// Copyright 2025 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <future>

#include "brpc/stream.h"
#include "spdlog/spdlog.h"

namespace spu::mpc::semi2k::ppmlac {

class StreamReader : public brpc::StreamInputHandler {
 public:
  enum class Status : int8_t {
    kNotFinished,
    kNormalFinished,
    kAbnormalFinished,
    kStreamFailed,
  };

  explicit StreamReader(const std::vector<size_t>& buf_lens)
      : buf_lens_(buf_lens) {
    future_finished_ = promise_finished_.get_future();
    future_closed_ = promise_closed_.get_future();
    buf_vec_.emplace_back();
  }

  int on_received_messages(brpc::StreamId id, butil::IOBuf* const messages[],
                           size_t size) override {
    for (size_t i = 0; i < size; ++i) {
      const auto& message = messages[i];
      SPDLOG_DEBUG("stream {} receive buf size: {}", id, message->size());
      if (status_ != Status::kNotFinished) {
        SPDLOG_ERROR("received redundant message");
        return 1;
      }
      butil::IOBuf& buf = buf_vec_.back();
      size_t max_buf_size = buf_lens_[buf_vec_.size() - 1];
      buf.append(message->movable());
      if (buf.size() > max_buf_size) {
        SPDLOG_ERROR("received overlong message, {} > {}", buf.size(),
                     max_buf_size);
        status_ = Status::kAbnormalFinished;
        promise_finished_.set_value(status_);
      } else if (buf.size() == max_buf_size) {
        if (buf_vec_.size() == buf_lens_.size()) {
          status_ = Status::kNormalFinished;
          promise_finished_.set_value(status_);
        } else {
          buf_vec_.emplace_back();
        }
      }
    }
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    SPDLOG_WARN("Stream {} idle timeout", id);
  }

  void on_closed(brpc::StreamId id) override {
    SPDLOG_DEBUG("Stream {} closed", id);
    promise_closed_.set_value();
  }

  void on_failed(brpc::StreamId id, int error_code,
                 const std::string& error_text) override {
    SPDLOG_ERROR("Stream {} failed, error_code: {}, error_text: {}", id,
                 error_code, error_text);
    promise_finished_.set_value(Status::kStreamFailed);
  }

  const std::vector<butil::IOBuf>& GetBufVecRef() const {
    SPU_ENFORCE(status_ == Status::kNormalFinished);
    return buf_vec_;
  }

  Status WaitFinished() { return future_finished_.get(); }
  void WaitClosed() { future_closed_.wait(); }

 private:
  std::vector<butil::IOBuf> buf_vec_;
  std::vector<size_t> buf_lens_;
  Status status_ = Status::kNotFinished;
  std::promise<Status> promise_finished_;
  std::promise<void> promise_closed_;
  std::future<Status> future_finished_;
  std::future<void> future_closed_;
};

}  // namespace spu::mpc::semi2k::ppmlac
