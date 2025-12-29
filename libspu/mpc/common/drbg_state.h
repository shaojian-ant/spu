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

#include "libspu/mpc/common/prg_state.h"

namespace spu::mpc {

class DrbgState : public PrgState {
  std::unique_ptr<yacl::crypto::Drbg> self_drbg_;
  std::unique_ptr<yacl::crypto::Drbg> next_drbg_;

 public:
  explicit DrbgState(const std::shared_ptr<yacl::link::Context>& lctx);

  NdArrayRef genPubl(FieldType field, const Shape& shape) override;

  std::pair<NdArrayRef, NdArrayRef> genPrssPair(FieldType field,
                                                const Shape& shape,
                                                GenPrssCtrl ctrl) override;
};

}  // namespace spu::mpc