#include "Backend/Backend.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#include "Profiler/Cupti/CuptiProfiler.h"
#include "Profiler/Instrumentation/InstrumentationProfiler.h"
#include "Profiler/RocprofSDK/RocprofSDKProfiler.h"
#include "Profiler/Roctracer/RoctracerProfiler.h"
#include "Runtime/CudaRuntime.h"
#include "Runtime/HipRuntime.h"
#include <algorithm>
#include <vector>

namespace proton {

namespace {

// Backends registered at runtime via registerBackend().
std::vector<BackendRegistration> &dynamicBackendRegistrations() {
  static std::vector<BackendRegistration> registrations;
  return registrations;
}

// Visits every registration: the compile-time ones first, then the runtime
// ones, so a backend linked into Proton keeps priority over one registered
// later.
template <typename Fn> void forEachBackendRegistration(Fn &&fn) {
  for (const auto &backend : getBackendRegistrations())
    fn(backend);
  for (const auto &backend : dynamicBackendRegistrations())
    fn(backend);
}

} // namespace

std::optional<DeviceType> allocateExternalDeviceType() {
  static size_t nextSlot = 0;
  constexpr auto firstExternal = static_cast<size_t>(DeviceType::EXTERNAL_0);
  const auto slot = firstExternal + nextSlot;
  if (slot >= static_cast<size_t>(DeviceType::COUNT))
    return std::nullopt;
  ++nextSlot;
  return static_cast<DeviceType>(slot);
}

bool registerBackend(BackendRegistration registration) {
  if (const auto &profiler = registration.getProfiler()) {
    const auto names = getRegisteredProfilerNames();
    if (std::find(names.begin(), names.end(), profiler->getName()) !=
        names.end())
      return false;
  }
  dynamicBackendRegistrations().push_back(std::move(registration));
  return true;
}

const std::vector<ProfilerRegistration> getProfilerRegistrations() {
  std::vector<ProfilerRegistration> registeredProfilers = {
      {"cupti", "cuda", []() { return &CuptiProfiler::instance(); }},
      {"rocprofiler", "hip", []() { return &RocprofSDKProfiler::instance(); }},
      {"roctracer", {}, []() { return &RoctracerProfiler::instance(); }},
      {"instrumentation",
       {},
       []() { return &InstrumentationProfiler::instance(); }},
  };
  forEachBackendRegistration([&](const BackendRegistration &backend) {
    if (const auto &profiler = backend.getProfiler())
      registeredProfilers.push_back(*profiler);
  });
  return registeredProfilers;
}

const std::vector<DeviceRegistration> getDeviceRegistrations() {
  std::vector<DeviceRegistration> registeredDevices = {
      {"CUDA", DeviceType::CUDA,
       [](uint64_t index) { return cuda::getDevice(index); }},
      {"HIP", DeviceType::HIP,
       [](uint64_t index) { return hip::getDevice(index); }},
  };
  forEachBackendRegistration([&](const BackendRegistration &backend) {
    if (const auto &device = backend.getDevice())
      registeredDevices.push_back(*device);
  });
  return registeredDevices;
}

const std::vector<RuntimeRegistration> getRuntimeRegistrations() {
  std::vector<RuntimeRegistration> registeredRuntimes = {
      {"CUDA", []() { return &CudaRuntime::instance(); }},
      {"HIP", []() { return &HipRuntime::instance(); }},
  };
  forEachBackendRegistration([&](const BackendRegistration &backend) {
    if (const auto &runtime = backend.getRuntime())
      registeredRuntimes.push_back(*runtime);
  });
  return registeredRuntimes;
}

const std::vector<std::string> getRegisteredProfilerNames() {
  const auto profilers = getProfilerRegistrations();
  std::vector<std::string> profilerNames(profilers.size());
  std::transform(
      profilers.begin(), profilers.end(), profilerNames.begin(),
      [](const ProfilerRegistration &entry) { return entry.getName(); });
  return profilerNames;
}

const std::optional<std::string>
getProfilerForTritonBackend(const std::string &tritonBackend) {
  const auto profilers = getProfilerRegistrations();
  auto itr = std::find_if(profilers.begin(), profilers.end(),
                          [&](const ProfilerRegistration &entry) {
                            return proton::toLower(tritonBackend) ==
                                   proton::toLower(
                                       entry.getTritonBackend().value_or(""));
                          });
  if (itr == profilers.end()) {
    return {};
  }
  return itr->getName();
}

} // namespace proton
