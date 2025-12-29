// Copyright 2024 Ant Group Co., Ltd.
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

#include "yacl/crypto/utils/drbg/drbg.h"

#include "libspu/mpc/common/drbg_tensor.h"
#include "libspu/mpc/semi2k/beaver/beaver_ttp.h"

namespace spu::mpc::semi2k {

class BeaverTtpDrbg final : public BeaverTtp {
 public:
  explicit BeaverTtpDrbg(std::shared_ptr<yacl::link::Context> lctx,
                         Options ops);

  ~BeaverTtpDrbg() override = default;

  Triple Mul(FieldType field, const Shape& shape) override;

  Triple And(FieldType field, const Shape& shape) override;

  Pair Trunc(FieldType field, const Shape& shape, size_t bits) override;

  Triple TruncPr(FieldType field, const Shape& shape, size_t bits) override;

  NdArrayRef RandBit(FieldType field, const Shape& shape) override;

  Pair PermPair(FieldType field, const Shape& shape, size_t perm_rank,
                absl::Span<const int64_t> perm_vec) override;

  std::unique_ptr<Beaver> Spawn() override;

  Pair Eqz(FieldType field, const Shape& shape) override;

 private:
  NdArrayRef CreateArray(FieldType field, int64_t m, int64_t n,
                         PrgArrayDesc& desc) override;

  std::unique_ptr<yacl::crypto::Drbg> drbg_;
};

}  // namespace spu::mpc::semi2k
