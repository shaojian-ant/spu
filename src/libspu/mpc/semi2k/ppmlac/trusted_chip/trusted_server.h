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

#include <optional>

#include "brpc/server.h"
#include "yacl/base/buffer.h"

namespace spu::mpc::semi2k::ppmlac {

struct TrustedServerOptions {
  int32_t port;
  // asym_crypto_schema: support ["SM2"]
  std::string asym_crypto_schema;
  yacl::Buffer public_key;
  yacl::Buffer private_key;
  std::optional<brpc::ServerSSLOptions> brpc_ssl_options;
};

std::unique_ptr<brpc::Server> RunServer(const TrustedServerOptions& options);
int RunUntilAskedToQuit(const TrustedServerOptions& options);

}  // namespace spu::mpc::semi2k::ppmlac
