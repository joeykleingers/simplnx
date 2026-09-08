# End-to-end acceptance for the storage-failure propagation chain in the command line runner.
#
# The pipeline creates a 32 MiB int32 array with the "HDF5-OOC" data format and then asks
# Compute Array Statistics to read every tuple of it. SIMPLNX_OOC_FAULT_INJECT tells the
# out-of-core layer to truncate that array's session file once, right after the pipeline flushes
# the data structure at the end of the creating filter. The read in the next filter therefore
# meets a genuinely damaged file.
#
# The run must fail loudly and survive: a normal process exit with a non-zero code, output carrying
# the reading filter's bulk-read failure for 'Data/Array', and an out-of-core store error code.
# Anything quieter means a storage error was swallowed somewhere between the store and the user.
#
# This script runs only where the fault seam is compiled in; SimplnxOoc's OOC.cmake registers it there.
#
# Set VERBOSE to have a passing run print the captured output.

if(NOT DEFINED NXRUNNER_EXECUTABLE)
  message(FATAL_ERROR "NXRUNNER_EXECUTABLE was not provided")
endif()
if(NOT DEFINED PIPELINE_FILE)
  message(FATAL_ERROR "PIPELINE_FILE was not provided")
endif()
if(NOT EXISTS "${PIPELINE_FILE}")
  message(FATAL_ERROR "PIPELINE_FILE does not exist: ${PIPELINE_FILE}")
endif()

set(ENV{SIMPLNX_OOC_FAULT_INJECT} "truncate-session-after-first-flush")

execute_process(
  COMMAND "${NXRUNNER_EXECUTABLE}" --execute "${PIPELINE_FILE}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
)

unset(ENV{SIMPLNX_OOC_FAULT_INJECT})

set(out "${run_stdout}${run_stderr}")

# execute_process() reports a signal or a failure to launch as text instead of a number. The storage
# error must reach the user through the normal reporting path, so a crashed or unlaunched process is
# a failure of this test even though it is also "not zero".
if(NOT run_result MATCHES "^[0-9]+$")
  message(FATAL_ERROR "nxrunner did not exit normally (${run_result}); the storage failure must not terminate the process:\n${out}")
endif()

if(run_result EQUAL 0)
  message(FATAL_ERROR "nxrunner reported success although the OOC session file was truncated:\n${out}")
endif()

# The fault line itself. Every filter announces its human name while it runs, so a name match alone
# would pass on any failure; the message the reading filter builds around the store error is what
# tells the user which array could not be read.
if(NOT out MATCHES "bulk read failed for array 'Data/Array'")
  message(FATAL_ERROR "nxrunner output does not report the failed bulk read of 'Data/Array':\n${out}")
endif()

# The reported code must come from the out-of-core store family (-6030 through -6039) so the message
# carries the storage error rather than a preflight or parameter error that happens to mention the array.
if(NOT out MATCHES "-603[0-9]")
  message(FATAL_ERROR "nxrunner output does not carry an out-of-core store error code:\n${out}")
endif()

if(VERBOSE)
  message(STATUS "${out}")
endif()
message(STATUS "nxrunner exited with ${run_result} and reported the storage failure to the user.")
