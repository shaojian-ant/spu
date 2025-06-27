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

#include "libspu/mpc/semi2k/ppmlac/ppmlac_factory.h"

#include "libspu/mpc/semi2k/ppmlac/ppmlac_receiver.h"
#include "libspu/mpc/semi2k/ppmlac/ppmlac_sender.h"

namespace spu::mpc::semi2k::ppmlac {

std::unique_ptr<PPMLAC> PPMLACFactory::CreatePPMLAC(
    const PPMLAC::Options& options,
    const std::shared_ptr<yacl::link::Context>& lctx) {
  if (options.recv_rank == lctx->Rank()) {
    const auto& receiver_options =
        dynamic_cast<const ReceiverPPMLAC::Options&>(options);
    return std::make_unique<ReceiverPPMLAC>(lctx, receiver_options);
  } else {
    const auto& sender_options =
        dynamic_cast<const SenderPPMLAC::Options&>(options);
    return std::make_unique<SenderPPMLAC>(lctx, sender_options);
  }
}

}  // namespace spu::mpc::semi2k::ppmlac
