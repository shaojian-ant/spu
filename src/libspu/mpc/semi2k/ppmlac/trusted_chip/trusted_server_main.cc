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

#include "yacl/crypto/key_utils.h"

#include "libspu/mpc/semi2k/ppmlac/trusted_chip/trusted_server.h"

namespace ppmlac_config {
DEFINE_int32(port, 16866, "TCP Port of this server");
}

int main(int argc, char* argv[]) {
  spu::mpc::semi2k::ppmlac::TrustedServerOptions options;
  options.port = ppmlac_config::FLAGS_port;
  options.asym_crypto_schema = "sm2";
  auto [pk, sk] = yacl::crypto::GenSm2KeyPairToPemBuf();
  options.public_key = pk;
  options.private_key = sk;
  return RunUntilAskedToQuit(options);
}
