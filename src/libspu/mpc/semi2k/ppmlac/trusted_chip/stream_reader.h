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

#include "brpc/stream.h"
#include "spdlog/spdlog.h"

namespace spu::mpc::semi2k::ppmlac {

class StreamReader : public brpc::StreamInputHandler {
 public:
  StreamReader(int32_t num_buf, size_t buf_len) {
    buf_vec_.resize(num_buf);
    buf_len_ = buf_len;
  }

  int on_received_messages(brpc::StreamId id, butil::IOBuf* const messages[],
                           size_t size) override {
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    SPDLOG_WARN("Stream {} idle timeout", id);
  }

  void on_closed(brpc::StreamId id) override {
    SPDLOG_DEBUG("Stream {} closed", id);
  }

  void on_failed(brpc::StreamId id, int error_code,
                 const std::string& error_text) override {
    SPDLOG_ERROR("Stream {} failed, error_code: {}, error_text: {}", id,
                 error_code, error_text);
  }

 private:
  std::vector<butil::IOBuf> buf_vec_;
  size_t buf_len_;
};

}  // namespace spu::mpc::semi2k::ppmlac
