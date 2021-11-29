add_llvm_library(LLVMOPUDesc
  OPUAsmBackend.cpp
  OPUELFObjectWriter.cpp
  OPUELFStreamer.cpp
  OPUInstPrinter.cpp
  OPUMCAsmInfo.cpp
  OPUMCCodeEmitter.cpp
  OPUMCExpr.cpp
  OPUMCTargetDesc.cpp
  OPUTargetStreamer.cpp

  LINK_COMPONENTS
  Core
  MC
  OPUInfo
  OPUUtils
  Support
  BinaryFormat

  #ADD_TO_COMPONENT
  #OPU
)

add_dependencies(LLVMOPUDesc LLVMOPUUtils)
