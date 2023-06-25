#include "gtest/gtest.h"
#include "yacl/link/link.h"

#include "libspu/mpc/object.h"

namespace spu::mpc::test {

using CreateObjectFn = std::function<std::unique_ptr<Object>(
    const RuntimeConfig& conf,
    const std::shared_ptr<yacl::link::Context>& lctx)>;

class ArithmeticTest : public ::testing::TestWithParam<
                           std::tuple<CreateObjectFn, RuntimeConfig, size_t>> {
};

}  // namespace spu::mpc::test