include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_library(LLVMOPUDisassembler
  OPUDisassembler.cpp

  LINK_COMPONENTS
  OPUDesc
  OPUInfo
  OPUUtils
  MC
  MCDisassembler
  Support

  #ADD_TO_COMPONENT
  #OPU

  )

add_dependencies(LLVMOPUDisassembler OPUCommonTableGen LLVMOPUUtils)
