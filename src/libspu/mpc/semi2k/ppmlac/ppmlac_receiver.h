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

#include "brpc/channel.h"

#include "libspu/mpc/semi2k/ppmlac/ppmlac.h"

namespace spu::mpc::semi2k::ppmlac {

class ReceiverPPMLAC : public PPMLAC {
 public:
  struct Options : PPMLAC::Options {
    std::string trusted_server_host;
    std::string brpc_channel_protocol = "baidu_std";
    int32_t brpc_connect_timeout_ms = 10 * 1000;
    int32_t brpc_timeout_ms = 300 * 1000;
    int32_t brpc_max_retry = 5;
    std::optional<brpc::ChannelSSLOptions> brpc_ssl_options;
  };

  ReceiverPPMLAC(const std::shared_ptr<yacl::link::Context>& lctx,
                 const Options& options)
      : PPMLAC(lctx) {
    InitChannel(options);
  }

  ~ReceiverPPMLAC() override = default;

  void Initialize() override;

  NdArrayRef Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                 const NdArrayRef& y) override;

  NdArrayRef Square(KernelEvalContext* ctx, const NdArrayRef& x) override;

  NdArrayRef MatMul(KernelEvalContext* ctx, const NdArrayRef& x,
                 const NdArrayRef& y) override;

  NdArrayRef And(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) override;

  NdArrayRef Trunc(KernelEvalContext* ctx, const NdArrayRef& x,
                   size_t bits) override;

  NdArrayRef B2A(KernelEvalContext* ctx, const NdArrayRef& x) override;

  NdArrayRef Eqz(KernelEvalContext* ctx, const NdArrayRef& z) override;

  NdArrayRef Perm(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& perm, size_t perm_rank) override;

 private:
  void InitChannel(const Options& options);

  mutable brpc::Channel channel_;  // TODO: mutable?
};

}  // namespace spu::mpc::semi2k::ppmlac
