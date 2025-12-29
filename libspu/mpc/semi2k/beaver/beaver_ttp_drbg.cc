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

#include "libspu/mpc/semi2k/beaver/beaver_ttp_drbg.h"

namespace spu::mpc::semi2k {

BeaverTtpDrbg::BeaverTtpDrbg(std::shared_ptr<yacl::link::Context> lctx,
                             Options ops)
    : BeaverTtp(std::move(lctx), std::move(ops)) {
  drbg_ = yacl::crypto::DrbgFactory::Instance().Create(
      "IC-HASH-DRBG", yacl::ArgLib = "Interconnection");

  drbg_->SetSeed(seed_);
}

Beaver::Triple BeaverTtpDrbg::Mul(FieldType field, const Shape& shape) {
  SPU_THROW("Unimplemented method");
}

Beaver::Triple BeaverTtpDrbg::And(FieldType field, const Shape& shape) {
  SPU_THROW("Unimplemented method");
}

Beaver::Pair BeaverTtpDrbg::Trunc(FieldType field, const Shape& shape,
                                  size_t bits) {
  SPU_THROW("Unimplemented method");
}

Beaver::Triple BeaverTtpDrbg::TruncPr(FieldType field, const Shape& shape,
                                      size_t bits) {
  SPU_THROW("Unimplemented method");
}

NdArrayRef BeaverTtpDrbg::RandBit(FieldType field, const Shape& shape) {
  SPU_THROW("Unimplemented method");
}

Beaver::Pair BeaverTtpDrbg::PermPair(FieldType field, const Shape& shape,
                                     size_t perm_rank,
                                     absl::Span<const int64_t> perm_vec) {
  SPU_THROW("Unimplemented method");
}

std::unique_ptr<Beaver> BeaverTtpDrbg::Spawn() {
  SPU_THROW("Unimplemented method");
}

Beaver::Pair BeaverTtpDrbg::Eqz(FieldType field, const Shape& shape) {
  SPU_THROW("Unimplemented method");
}

NdArrayRef BeaverTtpDrbg::CreateArray(FieldType field, int64_t m, int64_t n,
                                      PrgArrayDesc& desc) {
  return DrbgCreateArray(field, {m, n}, *drbg_, counter_, desc);
}

}  // namespace spu::mpc::semi2k