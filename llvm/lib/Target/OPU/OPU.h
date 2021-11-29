//===-- OPU.h - MachineFunction passes hw codegen --------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPU_H
#define LLVM_LIB_TARGET_OPU_OPU_H


#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/IntrinsicsOPU.h"

namespace llvm {

class OPUTargetMachine;
class FunctionPass;
class ModulePass;
class Pass;
class Target;
class TargetMachine;
class TargetOptions;
class PassRegistry;
class Module;

FunctionPass *createOPUISelDag(
  TargetMachine *TM = nullptr,
  CodeGenOpt::Level OptLevel = CodeGenOpt::Default);

FunctionPass *createOPUAnnotateUniformValues();
FunctionPass *createOPUAnnotateControlFlowPass();
FunctionPass *createOPULowerArgsPass(const OPUTargetMachine *TM, bool EnableRestrict);
FunctionPass *createOPUInsertWaitcntsPass();
FunctionPass *createOPUCodeGenPreparePass();
FunctionPass *createOPUAtomicOptimizerPass();
ModulePass *createOPULowerIntrinsicsPass();
ModulePass *createOPUAlwaysInlinePass(bool GlobalOpt = true, bool isSimt = false);
FunctionPass *createOPUOptimizeExecMaskingPreRAPass();
FunctionPass *createOPULowerI1CopiesPass();
FunctionPass *createOPUInsertI1CopiesPass();

ImmutablePass *createOPUAAWrapperPass();
void initializeOPUAAWrapperPassPass(PassRegistry&);
ImmutablePass *createOPUExternalAAWrapperPass();
void initializeOPUExternalAAWrapperPass(PassRegistry&);

void initializeOPUDAGToDAGISelPass(PassRegistry&);

void initializeOPUUnifyDivergentExitNodesPass(PassRegistry&);
extern char &OPUUnifyDivergentExitNodesID;

void initializeOPUAnnotateUniformValuesPass(PassRegistry&);
extern char &OPUAnnotateUniformValuesPassID;

void initializeOPUAnnotateControlFlowPass(PassRegistry&);
extern char &OPUAnnotateControlFlowPassID;

void initializeOPULowerControlFlowPass(PassRegistry &);
extern char &OPULowerControlFlowID;

void initializeOPUInsertSkipsPass(PassRegistry &);
extern char &OPUInsertSkipsPassID;

void initializeOPUArgumentInfoPass(PassRegistry &);

void initializeOPUInsertWaitcntsPass(PassRegistry&);
extern char &OPUInsertWaitcntsID;

void initializeOPUPreRAFusionPass(PassRegistry&);
extern char &OPUPreRAFusionPassID;

void initializeOPUCodeGenPreparePass(PassRegistry&);
extern char &OPUCodeGenPrepareID;

void initializeOPUAtomicOptimizerPass(PassRegistry &);
extern char &OPUAtomicOptimizerID;

void initializeOPULowerIntrinsicsPass(PassRegistry &);
extern char &OPULowerIntrinsicsID;

void initializeOPUAlwaysInlinePass(PassRegistry&);

void initializeOPUOptimizeExecMaskingPreRAPass(PassRegistry&);
extern char &OPUOptimizeExecMaskingPreRAID;

void initializeOPUOptimizeExecMaskingPass(PassRegistry &);
extern char &OPUOptimizeExecMaskingID;

void initializeOPULowerI1CopiesPass(PassRegistry &);
extern char &OPULowerI1CopiesID;

void initializeOPUInsertI1CopiesPass(PassRegistry &);
extern char &OPUInsertI1CopiesID;

void initializeOPULowerSGPRSpillsPass(PassRegistry &);
extern char &OPULowerSGPRSpillsID;

Pass *createOPUAnnotateKernelFeaturesPass();
void initializeOPUAnnotateKernelFeaturesPass(PassRegistry &);
extern char &OPUAnnotateKernelFeaturesID;

FunctionPass *createOPUAsyncCopyAnalysisPass();
void initializeOPUAsyncCopyAnalysisPass(PassRegistry &);

//FunctionPass *createOPUReorderBlocksPass();
void initializeOPUReorderBlocksPass(PassRegistry &);
extern char &OPUReorderBlocksPassID;

FunctionPass *createOPULowerAllocaPass();
void initializeOPULowerAllocaPass(PassRegistry&);

FunctionPass *createOPUAnnotateModeRegisterPass();
void initializeOPUAnnotateModeRegisterPass(PassRegistry&);
extern char &OPUAnnotateModeRegisterPassID;

FunctionPass *createOPUForceSetCachePolicyPass();
void initializeOPUForceSetCachePolicyPass(PassRegistry&);
extern char &OPUForceSetCachePolicyPassID;

FunctionPass *createOPUAnnotateSmemsPass();
void initializeOPUAnnotateSmemsPass(PassRegistry&);
extern char &OPUAnnotateSmemsPassID;

FunctionPass *createOPUMemAnalysisPass();
void initializeOPUMemAnalysisPass(PassRegistry&);

FunctionPass *createOPUDepResolverPass();
void initializeOPUDepResolverPass(PassRegistry&);
extern char &OPUDepResolverPassID;

FunctionPass *createOPUAnnotateX0PatternsPass();
void initializeOPUAnnotateX0PatternsPass(PassRegistry&);
extern char &OPUAnnotateX0PatternsPassID;

FunctionPass *createOPUMemoryLegalizerPass();
void initializeOPUMemoryLegalizerPass(PassRegistry&);
extern char &OPUMemoryLegalizerID;

FunctionPass *createOPULoadStoreOptimizerPass();
void initializeOPULoadStoreOptimizerPass(PassRegistry &);
extern char &OPULoadStoreOptimizerID;

FunctionPass *createOPUAtomicSpliterPass();
void initializeOPUAtomicSpliterPass(PassRegistry &);
extern char &OPUAtomicSpliterID;

FunctionPass *createOPUMMAExtendLiveRangesPass();
void initializeOPUMMAExtendLiveRangesPass(PassRegistry &);
extern char &OPUMMAExtendLiveRangesID;

FunctionPass *createOPUFixUninitializesPass();
void initializeOPUFixUninitializesPass(PassRegistry &);
extern char &OPUFixUninitializesID;

FunctionPass *createOPUPromoteAlloca();
void initializeOPUPromoteAllocaPass(PassRegistry&);
extern char &OPUPromoteAllocaID;

// FunctionPass *createOPUUnifyDivergentBackEdgesPass();
void initializeOPUUnifyDivergentBackEdgesPass(PassRegistry &);
extern char &OPUUnifyDivergentBackEdgesID;

void initializeOPURecordUndefRegsPass(PassRegistry &);
extern char &OPURecordUndefRegsID;

FunctionPass *createOPUFoldOperandsPass();
void initializeOPUFoldOperandsPass(PassRegistry &);
extern char &OPUFoldOperandsID;

FunctionPass *createOPUFoldLogicOpPass();
void initializeOPUFoldLogicOpPass(PassRegistry &);
extern char &OPUFoldLogicOpID;

FunctionPass *createOPUPreAllocateFoldPass();
void initializeOPUPreAllocateFoldPass(PassRegistry &);
extern char &OPUPreAllocateFoldID;

void initializeOPUFixVGPRCopiesPass(PassRegistry &);
extern char &OPUFixVGPRCopiesID;

} // end namespace LLVM

// FIXME schi below copied from AMDGPU.h
/// OpenCL uses address spaces to differentiate between
/// various memory regions on the hardware. On the CPU
/// all of the address spaces point to the same memory,
/// however on the GPU, each address space points to
/// a separate piece of memory that is unique from other
/// memory locations.
namespace OPUAS {
  enum : unsigned {
    // The maximum value for flat, generic, local, private, constant and region.
    MAX_AMDGPU_ADDRESS = 7,

    FLAT_ADDRESS = 0,     ///< Address space for flat memory.
    GLOBAL_ADDRESS = 1,   ///< Address space for global memory (RAT0, VTX0).
    REGION_ADDRESS = 2,   ///< Address space for region memory. (GDS)

    CONSTANT_ADDRESS = 4, ///< Address space for constant memory (VTX2).
    LOCAL_ADDRESS = 3,    ///< Address space for local memory.
    PRIVATE_ADDRESS = 5,  ///< Address space for private memory.

    CONSTANT_ADDRESS_32BIT = 6, ///< Address space for 32-bit constant memory.

    BUFFER_FAT_POINTER = 7, ///< Address space for 160-bit buffer fat pointers.

    /// Address space for direct addressible parameter memory (CONST0).
    PARAM_D_ADDRESS = 6,
    /// Address space for indirect addressible parameter memory (VTX1).
    PARAM_I_ADDRESS = 7,

    // Do not re-order the CONSTANT_BUFFER_* enums.  Several places depend on
    // this order to be able to dynamically index a constant buffer, for
    // example:
    //
    // ConstantBufferAS = CONSTANT_BUFFER_0 + CBIdx

    CONSTANT_BUFFER_0 = 8,
    CONSTANT_BUFFER_1 = 9,
    CONSTANT_BUFFER_2 = 10,
    CONSTANT_BUFFER_3 = 11,
    CONSTANT_BUFFER_4 = 12,
    CONSTANT_BUFFER_5 = 13,
    CONSTANT_BUFFER_6 = 14,
    CONSTANT_BUFFER_7 = 15,
    CONSTANT_BUFFER_8 = 16,
    CONSTANT_BUFFER_9 = 17,
    CONSTANT_BUFFER_10 = 18,
    CONSTANT_BUFFER_11 = 19,
    CONSTANT_BUFFER_12 = 20,
    CONSTANT_BUFFER_13 = 21,
    CONSTANT_BUFFER_14 = 22,
    CONSTANT_BUFFER_15 = 23,

    // Some places use this if the address space can't be determined.
    UNKNOWN_ADDRESS_SPACE = ~0u,
  };
}

#endif
