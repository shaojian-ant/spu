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

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/common/prg_tensor.h"

namespace spu::mpc {

using DrbgPtr = std::unique_ptr<yacl::crypto::Drbg>;

constexpr char kIcBeaverSessionPrefix[] = "interconnection";

inline NdArrayRef DrbgCreateArray(FieldType field, const Shape &shape,
                                  yacl::crypto::Drbg &drbg, PrgCounter &counter,
                                  PrgArrayDesc &desc) {
  desc = {Shape(shape.begin(), shape.end()), field, counter};

  NdArrayRef ret(makeType<RingTy>(field), shape);
  drbg.Fill(ret.data<char>(), ret.buf()->size());
  counter += ret.buf()->size();

  return ret;
}

inline NdArrayRef DrbgReplayArray(yacl::crypto::Drbg &drbg,
                                  const PrgArrayDesc &desc) {
  NdArrayRef ret(makeType<RingTy>(desc.field), desc.shape);
  drbg.Fill(ret.data<char>(), ret.buf()->size());

  return ret;
}

}  // namespace spu::mpc
