# TraceData is tested through its own sources rather than the ProtonData library: the
# library also carries TreeData, whose device-type lookups pull in every profiler.
add_triton_ut(
  NAME TraceData
  SRCS TraceDataTest.cpp
       "${PROTON_SRC_DIR}/lib/Context/Context.cpp"
       "${PROTON_SRC_DIR}/lib/Data/Data.cpp"
       "${PROTON_SRC_DIR}/lib/Data/Metric.cpp"
       "${PROTON_SRC_DIR}/lib/Data/TraceData.cpp"
       "${PROTON_SRC_DIR}/lib/Data/Dump/ChromeTrace.cpp"
       "${PROTON_SRC_DIR}/lib/Data/Dump/TraceDataDump.cpp"
  LIBS ProtonUtility ProtonTraceDataIO
)

target_include_directories(TraceData
  PRIVATE
    "${JSON_INCLUDE_DIR}"
    "${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
    "${PROTON_SRC_DIR}/lib/Data"
)
