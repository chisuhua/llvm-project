//===- AssumeBundleBuilder.h - utils to build assume bundles ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contain tools to preserve informations. They should be used before
// performing a transformation that may move and delete instructions as those
// transformation may destroy or worsen information that can be derived from the
// IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_IRBUILDER_H
#define LLVM_TRANSFORMS_UTILS_IRBUILDER_H

#include "llvm/IR/IRBuilder.h"

namespace llvm {

Value *emitOPUPrintfCall(IRBuilder<> &Builder, ArrayRef<Value *> Args);

}

#endif
