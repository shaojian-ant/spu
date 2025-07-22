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

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/prng.h"

#include "yacl/crypto/tools/prg.h"

#include "libspu/mpc/utils/permute.h"

namespace spu::mpc::semi2k::ppmlac {

// Fill the output with generated randomness
void PRNG::Fill(char* buf, size_t len) {
  counter_ = yacl::crypto::FillPRand(crypto_type_, seed_, 0, counter_,
                                     absl::MakeSpan(buf, len));
}

NdArrayRef PRNG::FillRing(const Type& eltype, const Shape& shape) {
  NdArrayRef ret(eltype, shape);
  Fill(ret.data<char>(), ret.buf()->size());
  return ret;
}

Index PRNG::GenRandomPerm(size_t numel) {
  return genRandomPerm(numel, seed_, &counter_);
}

}  // namespace spu::mpc::semi2k::ppmlac
