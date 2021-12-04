add_llvm_library(LLVMOPUAsmParser
  OPUAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  OPUDesc
  OPUInfo
  OPUUtils
  Support

  #ADD_TO_COMPONENT
  #OPU

  )

add_dependencies(LLVMOPUAsmParser LLVMOPUUtils)
