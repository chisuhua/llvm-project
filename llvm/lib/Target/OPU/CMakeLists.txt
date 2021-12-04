add_llvm_component_group(OPU)

set(LLVM_TARGET_DEFINITIONS OPU.td)

tablegen(LLVM OPUGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM OPUGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM OPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM OPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM OPUGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM OPUGenInstrInfo.inc -gen-instr-info)
#tablegen(LLVM OPUGenIntrinsicEnums.inc -gen-tgt-intrinsic-enums)
#tablegen(LLVM OPUGenIntrinsicImpl.inc -gen-tgt-intrinsic-impl)
tablegen(LLVM OPUGenMCCodeEmitter.inc -gen-emitter)
#tablegen(LLVM OPUGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM OPUGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM OPUGenRegisterInfo.inc -gen-register-info)
#tablegen(LLVM OPUGenSearchableTables.inc -gen-searchable-tables)
tablegen(LLVM OPUGenSubtargetInfo.inc -gen-subtarget)
#tablegen(LLVM OPUGenSystemOperands.inc -gen-searchable-tables)

set(LLVM_TARGET_DEFINITIONS OPUGISel.td)
tablegen(LLVM OPUGenGlobalISel.inc -gen-global-isel)
#tablegen(LLVM AMDGPUGenPreLegalizeGICombiner.inc -gen-global-isel-combiner
#              -combiners="AMDGPUPreLegalizerCombinerHelper")
#tablegen(LLVM AMDGPUGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
#              -combiners="AMDGPUPostLegalizerCombinerHelper")
#tablegen(LLVM AMDGPUGenRegBankGICombiner.inc -gen-global-isel-combiner
#              -combiners="AMDGPURegBankCombinerHelper")


add_public_tablegen_target(OPUCommonTableGen)

# TODO schi OPUOptimizeVSETVLUses.cpp is copied from rvv
add_llvm_target(OPUCodeGen
  OPUAliasAnalysis.cpp
  OPUAlwaysInlinePass.cpp
  OPUAnnotateControlFlow.cpp
  OPUAnnotateKernelFeatures.cpp
  #OPUAnnotateBlockDivergency.cpp
  OPUAnnotateModeRegister.cpp
  OPUAnnotateUniformValues.cpp
  OPUAnnotateCmems.cpp
  OPUArgumentInfo.cpp
  OPUAsmPrinter.cpp
  OPUAsyncCopyAnalysis.cpp
  OPUAtomicOptimizer.cpp
  OPUAtomicSpliter.cpp
  OPUCallingConv.cpp
  OPUCallLowering.cpp
  OPUCodeGenPrepare.cpp
  OPUDepResolver.cpp
  OPUEarlyCodeGenPrepare.cpp
  OPUEarlyIRTransform.cpp
  #OPUExpandPseudoInsts.cpp
  OPUFixUninitializes.cpp
  OPUFixVGPRCopies.cpp
  OPUFixSGPRCopies.cpp
  OPUFoldOperands.cpp
  OPUFoldLogicOp.cpp
  #OPUFormMemoryClauses.cpp
  OPUFrameLowering.cpp
  #OPUOptimizeVSETVLUses.cpp
  OPUHazardRecognizer.cpp
  OPUILPSched.cpp
  OPUInsertI1Copies.cpp
  OPUInsertSkips.cpp
  OPUInsertWaitcnts.cpp
  OPUInstrInfo.cpp
  OPUInstructionSelector.cpp
  OPUISelDAGToDAG.cpp
  OPUISelLowering.cpp
  OPUIterativeScheduler.cpp
  OPULegalizerInfo.cpp
  OPULoadStoreOptimizer.cpp
  OPULowerKernelArguments.cpp
  OPULowerAlloca.cpp
  OPULowerControlFlow.cpp
  OPULowerI1Copies.cpp
  OPULowerIntrinsics.cpp
  OPULowerSGPRSpills.cpp
  #OPULowerKernelAttributes.cpp
  OPULowerReconvergingControlFlow.cpp
  OPUMachineFunctionInfo.cpp
  OPUMachineModuleInfo.cpp
  OPUMemoryLegalizer.cpp
  OPUMinRegStrategy.cpp
  OPUMemAnalysis.cpp
  OPUMCInstLower.cpp
  #OPUMergeBaseOffset.cpp
  #OPUMacroFusion.cpp
  #OPUMachineScheduler.cpp
  OPUOptimizeExecMasking.cpp
  OPUOptimizeExecMaskingPreRA.cpp
  OPUOptimizeVGPRLiveRange.cpp
  OPUKernelMetaStreamer.cpp
  OPUPerfHintAnalysis.cpp
  #OPUPreAllocateWWMRegs.cpp
  OPUMetadataStreamer.cpp
  OPUPromoteAlloca.cpp
  #OPURegBankReassign.cpp
  OPURegisterBankInfo.cpp
  OPURegisterInfo.cpp
  OPURegPressure.cpp
  OPURemoveGuardRegs.cpp
  OPUReorderBlocks.cpp
  OPUResourceInfoAnalysis.cpp
  #OPURewriteOutArguments.cpp
  OPUSchedStrategy.cpp
  OPUSubtarget.cpp
  OPUTargetMachine.cpp
  OPUTargetObjectFile.cpp
  OPUTargetTransformInfo.cpp
  OPUUnifyDivergentBackEdges.cpp
  OPUUnifyDivergentExitNodes.cpp
  #OPUUnifyMetadata.cpp
  OPUWarpSyncInline.cpp
  OPUFixupVectorISel.cpp

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  Core
  IPO
  MC
  Passes
  OPUDesc
  OPUInfo
  OPUUtils
  Scalar
  SelectionDAG
  Support
  Target
  TransformUtils
  Vectorize
  GlobalISel
  BinaryFormat
  MIRParser

  ADD_TO_COMPONENT
  OPU
  )
# OPUNSAReassign.cpp
# OPUIntrinsicInfo.cpp

# FIXME
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
add_subdirectory(Utils)
