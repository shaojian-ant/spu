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

#include "libspu/mpc/semi2k/ppmlac/utils.h"

#include "spdlog/spdlog.h"

#include "libspu/core/prelude.h"

namespace spu::mpc::semi2k::ppmlac {

void SendMessage(brpc::StreamId stream_id, void* message, size_t size) {
  if (size == 0) {
    SPDLOG_WARN("Skip sending empty message");
    return;
  }

  butil::IOBuf buf;
  // buf.append_user_data(message, size, [](void*){});
  buf.append(message, size);

  const int ret = brpc::StreamWrite(stream_id, buf);
  SPU_ENFORCE_EQ(ret, 0, "brpc::StreamWrite failed with error code={}", ret);
}

}  // namespace spu::mpc::semi2k::ppmlac
