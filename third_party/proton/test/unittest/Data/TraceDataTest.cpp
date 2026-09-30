#include "Data/TraceData.h"
#include "Context/Context.h"
#include "Data/Metric.h"
#include "nlohmann/json.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace proton;

namespace {

// Records one kernel launched inside the scope "launch" and returns the chrome
// trace. With `intermediateEvent`, the kernel is recorded under an op event
// that has no CPU time range, the way a profiler does when its backend cannot
// name a kernel at the time the launch is observed.
json traceKernelInScope(bool intermediateEvent) {
  TraceData data("unused");
  ScopeInterface &scopes = data;
  const auto phase = data.getPhaseInfo().current;

  Scope launch("launch");
  scopes.enterScope(launch);
  auto parent = data.addOp(phase, Data::kRootEntryId, {Context("op")});
  auto kernel = intermediateEvent
                    ? data.addOp(phase, parent.id, {Context("kernel")})
                    : parent;
  kernel.upsertMetric(std::make_unique<KernelMetric>(
      /*startTime=*/1000, /*endTime=*/2000, /*invocations=*/1,
      /*deviceId=*/0, static_cast<uint64_t>(DeviceType::CUDA),
      /*streamId=*/0));
  scopes.exitScope(launch);

  return json::parse(data.toJsonString(phase));
}

// The kernel is linked to the CPU event that launched it by one flow arrow,
// which starts at that CPU event: same lane and timestamp as its slice.
void expectKernelAttributedTo(const json &trace, const std::string &launch) {
  std::vector<json> slices, starts, finishes;
  for (const auto &event : trace["traceEvents"]) {
    if (event.value("name", "") == launch && event.value("ph", "") == "X")
      slices.push_back(event);
    if (event.value("name", "") != "launch->kernel")
      continue;
    if (event.value("ph", "") == "s")
      starts.push_back(event);
    else if (event.value("ph", "") == "f")
      finishes.push_back(event);
  }
  ASSERT_EQ(slices.size(), 1u) << trace.dump(2);
  ASSERT_EQ(starts.size(), 1u) << trace.dump(2);
  ASSERT_EQ(finishes.size(), 1u) << trace.dump(2);
  EXPECT_EQ(starts[0]["id"], finishes[0]["id"]) << trace.dump(2);
  EXPECT_EQ(starts[0]["ts"], slices[0]["ts"]) << trace.dump(2);
  EXPECT_EQ(starts[0]["tid"], slices[0]["tid"]) << trace.dump(2);
}

} // namespace

TEST(TraceDataChromeTrace, KernelUnderScopeIsAttributedToScope) {
  expectKernelAttributedTo(traceKernelInScope(/*intermediateEvent=*/false),
                           "launch");
}

TEST(TraceDataChromeTrace, KernelUnderEventWithoutCpuTimesIsAttributedToScope) {
  expectKernelAttributedTo(traceKernelInScope(/*intermediateEvent=*/true),
                           "launch");
}
