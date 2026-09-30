#include "Device.h"
#include "Backend/Backend.h"
#include "DeviceType.h"

#include "Utility/Errors.h"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace proton {
namespace {

const DeviceRegistration getDeviceEntry(DeviceType type) {
  const auto devices = getDeviceRegistrations();
  auto itr = std::find_if(devices.begin(), devices.end(),
                          [&](const DeviceRegistration &entry) {
                            return type == entry.getDeviceType();
                          });
  if (itr == devices.end()) {
    throw makeInvalidArgument("DeviceType not supported");
  }
  return *itr;
}

} // namespace

Device getDevice(DeviceType type, uint64_t index) {
  return getDeviceEntry(type).getDevice()(index);
}

const std::string getDeviceTypeString(DeviceType type) {
  // Callers iterate every slot up to DeviceType::COUNT (see HatchetMsgPack),
  // which now includes reserved EXTERNAL_* slots that no backend has claimed.
  // Name those instead of throwing; a genuinely wrong lookup still surfaces
  // through getDevice().
  constexpr const char *kUnknownDeviceTypeName = "unknown";
  const auto devices = getDeviceRegistrations();
  auto itr = std::find_if(devices.begin(), devices.end(),
                          [&](const DeviceRegistration &entry) {
                            return type == entry.getDeviceType();
                          });
  if (itr == devices.end())
    return kUnknownDeviceTypeName;
  return itr->getName();
}

} // namespace proton
