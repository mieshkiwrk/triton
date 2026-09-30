// A stand-in for a Triton backend that ships as its own shared library, next to
// an already-built Proton, and therefore cannot be part of Proton's own build.
// It uses only Proton's exported registration entry points, so it also links
// and loads against a Proton built with hidden visibility.
#include "Backend/Backend.h"
#include "Device.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>

namespace {

// Every DeviceType value this library was handed, to detect a slot handed out
// twice.
std::set<proton::DeviceType> &claimedDeviceTypes() {
  static std::set<proton::DeviceType> claimed;
  return claimed;
}

// Returns false if Proton handed out a value outside the reserved range or one
// it had handed out before.
bool recordClaim(proton::DeviceType deviceType) {
  if (deviceType < proton::DeviceType::EXTERNAL_0 ||
      deviceType >= proton::DeviceType::COUNT)
    return false;
  return claimedDeviceTypes().insert(deviceType).second;
}

// Like a real backend, the fixture claims its device type once and reuses it
// for every registration it makes.
std::optional<proton::DeviceType> fixtureDeviceType() {
  static const auto deviceType = []() -> std::optional<proton::DeviceType> {
    auto claimed = proton::allocateExternalDeviceType();
    if (claimed && !recordClaim(*claimed))
      return std::nullopt;
    return claimed;
  }();
  return deviceType;
}

} // namespace

extern "C" {

// Registration is explicit rather than done from a loader constructor, so the
// test controls when it happens and can observe the result.
__attribute__((visibility("default"))) bool
proton_test_register_runtime_backend(const char *profilerName,
                                     const char *tritonBackend) {
  const auto deviceType = fixtureDeviceType();
  if (!deviceType)
    return false;
  return proton::registerBackend({
      proton::ProfilerRegistration{
          profilerName, tritonBackend,
          []() -> proton::Profiler * { return nullptr; }},
      proton::DeviceRegistration{"TEST_RUNTIME_DEVICE", *deviceType,
                                 [](uint64_t) { return proton::Device{}; }},
      proton::RuntimeRegistration{
          "TEST_RUNTIME_DEVICE", []() -> proton::Runtime * { return nullptr; }},
  });
}

// Claims device type slots until Proton reports that none is left. Returns how
// many this call claimed, or -1 if a claimed value was outside the reserved
// range, was handed out twice, or allocation never reported exhaustion.
__attribute__((visibility("default"))) int
proton_test_claim_device_types_until_exhausted() {
  const auto limit = static_cast<size_t>(proton::DeviceType::COUNT);
  for (size_t claimed = 0; claimed <= limit; ++claimed) {
    const auto deviceType = proton::allocateExternalDeviceType();
    if (!deviceType)
      return static_cast<int>(claimed);
    if (!recordClaim(*deviceType))
      return -1;
  }
  return -1;
}
}
