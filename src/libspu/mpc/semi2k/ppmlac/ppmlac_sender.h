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

#include "libspu/mpc/semi2k/ppmlac/ppmlac.h"
#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_chip.h"

namespace spu::mpc::semi2k::ppmlac {

class SenderPPMLAC : public PPMLAC {
 public:
  struct Options : PPMLAC::Options {
    std::string asym_crypto_schema;
    yacl::Buffer pub_key;   // public key of the trusted chip in PEM format
    yacl::Buffer priv_key;  // private key of the trusted chip in PEM format
  };

  SenderPPMLAC(const std::shared_ptr<yacl::link::Context>& lctx,
               const Options& options)
      : PPMLAC(lctx),
        recv_rank_(options.recv_rank),
        chip_(options.asym_crypto_schema, options.pub_key, options.priv_key) {}

  ~SenderPPMLAC() override = default;

  void Initialize() override;

  NdArrayRef Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                 const NdArrayRef& y) override;

  NdArrayRef Square(KernelEvalContext* ctx, const NdArrayRef& x) override;

  NdArrayRef MatMul(KernelEvalContext* ctx, const NdArrayRef& x,
                    const NdArrayRef& y) override;

  NdArrayRef And(KernelEvalContext* ctx, const NdArrayRef& lhs,
                 const NdArrayRef& rhs) override;

  NdArrayRef Trunc(KernelEvalContext* ctx, const NdArrayRef& x,
                   size_t bits) override;

  NdArrayRef B2A(KernelEvalContext* ctx, const NdArrayRef& x) override;

  NdArrayRef Eqz(KernelEvalContext* ctx, const NdArrayRef& z) override;

  NdArrayRef Perm(KernelEvalContext* ctx, const NdArrayRef& x,
                  const NdArrayRef& perm, size_t perm_rank) override;

 private:
  size_t recv_rank_;
  TrustedChip chip_;
};

}  // namespace spu::mpc::semi2k::ppmlac
