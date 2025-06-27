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

#include "yacl/base/int128.h"
#include "yacl/crypto/block_cipher/symmetric_crypto.h"

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::semi2k::ppmlac {

class PRNG {
  using CryptoType = yacl::crypto::SymmetricCrypto::CryptoType;

 public:
  explicit PRNG(uint128_t seed) : seed_(seed) {}

  // Fill the output with generated randomness
  void Fill(char* buf, size_t len);

  NdArrayRef FillRing(FieldType field, const Shape& shape);

 private:
  uint128_t seed_ = 0;
  uint64_t counter_ = 0;
  CryptoType crypto_type_ = CryptoType::AES128_ECB;
};

}  // namespace spu::mpc::semi2k::ppmlac
