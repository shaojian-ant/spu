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

#include "libspu/mpc/common/drbg_state.h"

#include <array>

#include "yacl/crypto/utils/drbg/drbg.h"
#include "yacl/crypto/utils/rand.h"
#include "yacl/utils/serialize.h"

namespace spu::mpc {

DrbgState::DrbgState(const std::shared_ptr<yacl::link::Context>& lctx) {
  // initialize self seed
  uint128_t self_seed = yacl::crypto::SecureRandSeed();

  // send seed to prev party, receive seed from next party
  constexpr char kCommTag[] = "Random:PRSS";
  lctx->SendAsync(lctx->PrevRank(), yacl::SerializeUint128(self_seed),
                  kCommTag);
  uint128_t next_seed =
      yacl::DeserializeUint128(lctx->Recv(lctx->NextRank(), kCommTag));

  // initialize DRBG
  self_drbg_ = yacl::crypto::DrbgFactory::Instance().Create(
      "IC-HASH-DRBG", yacl::ArgLib = "Interconnection");
  next_drbg_ = yacl::crypto::DrbgFactory::Instance().Create(
      "IC-HASH-DRBG", yacl::ArgLib = "Interconnection");

  self_drbg_->SetSeed(self_seed);
  next_drbg_->SetSeed(next_seed);
}

NdArrayRef DrbgState::genPubl(FieldType field, const Shape& shape) {
  SPU_THROW("not implemented");
}

std::pair<NdArrayRef, NdArrayRef> DrbgState::genPrssPair(FieldType field,
                                                         const Shape& shape,
                                                         GenPrssCtrl ctrl) {
  SPU_ENFORCE(ctrl == GenPrssCtrl::Both);
  const Type ty = makeType<RingTy>(field);

  NdArrayRef r_self(ty, shape);
  NdArrayRef r_next(ty, shape);

  self_drbg_->Fill(r_self.data<char>(), shape.numel() * ty.size());
  next_drbg_->Fill(r_next.data<char>(), shape.numel() * ty.size());

  return std::make_pair(r_self, r_next);
}

}  // namespace spu::mpc
