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

#include "yacl/link/context.h"

#include "libspu/core/context.h"
#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::semi2k::ppmlac {

class PPMLAC {
 public:
  struct Options {
    virtual ~Options() = default;
    size_t recv_rank;  // the rank of receiver in one-way communication
  };

  explicit PPMLAC(const std::shared_ptr<yacl::link::Context>& lctx)
      : lctx_(lctx) {}

  virtual ~PPMLAC() = default;

  virtual void Initialize() = 0;

  // TODO: need KernelEvalContext parameter?
  virtual NdArrayRef Mul(KernelEvalContext* ctx, const NdArrayRef& x,
                         const NdArrayRef& y) = 0;

 protected:
  std::shared_ptr<yacl::link::Context> lctx_;
};

}  // namespace spu::mpc::semi2k::ppmlac
