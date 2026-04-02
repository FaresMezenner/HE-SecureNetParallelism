# This file is configured by CMake automatically as DartConfiguration.tcl
# If you choose not to use CMake, this file may be hand configured, by
# filling in the required variables.


# Configuration directories and files
SourceDirectory: /home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/src
BuildDirectory: /home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/build/thirdparty/eigen/src

# Where to place the cost data store
CostDataFile: 

# Site is something like machine.domain, i.e. pragmatic.crd
Site: ws-alternative-01

# Build name is osname-revision-compiler, i.e. Linux-2.4.2-2smp-c++
BuildName: linux-6.14.0-1020-gcp-_-13.3.0-sse2-64bit

# Subprojects
LabelsForSubprojects: 

# Submission information
SubmitURL: http://manao.inria.fr/CDash/submit.php?project=Eigen+3.3
SubmitInactivityTimeout: 

# Dashboard start time
NightlyStartTime: 00:00:00 UTC

# Commands for the build/test/submit cycle
ConfigureCommand: "/usr/bin/cmake" "/home/lf_mezenner_udst_edu_qa/secure-parallel-training/HE-SecureNet/thirdparty/eigen/src"
MakeCommand: /usr/bin/cmake --build . --target buildtests --config "${CTEST_CONFIGURATION_TYPE}" -- -i 
DefaultCTestConfigurationType: Release

# version control
UpdateVersionOnly: 

# CVS options
# Default is "-d -P -A"
CVSCommand: 
CVSUpdateOptions: 

# Subversion options
SVNCommand: 
SVNOptions: 
SVNUpdateOptions: 

# Git options
GITCommand: /usr/bin/git
GITInitSubmodules: 
GITUpdateOptions: 
GITUpdateCustom: 

# Perforce options
P4Command: 
P4Client: 
P4Options: 
P4UpdateOptions: 
P4UpdateCustom: 

# Generic update command
UpdateCommand: /usr/bin/git
UpdateOptions: 
UpdateType: git

# Compiler info
Compiler: /usr/bin/c++
CompilerVersion: 13.3.0

# Dynamic analysis (MemCheck)
PurifyCommand: 
ValgrindCommand: 
ValgrindCommandOptions: 
DrMemoryCommand: 
DrMemoryCommandOptions: 
CudaSanitizerCommand: 
CudaSanitizerCommandOptions: 
MemoryCheckType: 
MemoryCheckSanitizerOptions: 
MemoryCheckCommand: /usr/local/cuda-12.8/bin/compute-sanitizer
MemoryCheckCommandOptions: 
MemoryCheckSuppressionFile: 

# Coverage
CoverageCommand: /usr/bin/gcov
CoverageExtraFlags: -l

# Testing options
# TimeOut is the amount of time in seconds to wait for processes
# to complete during testing.  After TimeOut seconds, the
# process will be summarily terminated.
# Currently set to 25 minutes
TimeOut: 1500

# During parallel testing CTest will not start a new test if doing
# so would cause the system load to exceed this value.
TestLoad: 

UseLaunchers: 
CurlOptions: 
# warning, if you add new options here that have to do with submit,
# you have to update cmCTestSubmitCommand.cxx

# For CTest submissions that timeout, these options
# specify behavior for retrying the submission
CTestSubmitRetryDelay: 5
CTestSubmitRetryCount: 3
