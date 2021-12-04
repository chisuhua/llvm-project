add_llvm_library(LLVMOPUUtils
  OPUBaseInfo.cpp
  OPUMatInt.cpp
  OPUAsmUtils.cpp
  OPUKernelCodeTUtils.cpp

  LINK_COMPONENTS
  Core
  MC
  BinaryFormat
  Support

  #ADD_TO_COMPONENT
  #OPU
  )
