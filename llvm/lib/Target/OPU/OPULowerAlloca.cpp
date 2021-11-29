//  from:
//      %A = alloca i32
//      store i32 0, i32* %A;  emits st.u32
//
//  to:
//      %A = alloca i32
//      %Local = addrspacecast i32* %A to i32 addrspace(5)*
//      %Generic = addrspacecast i32 addrspace(5)* %A to i32*
//      store i32 0, i32 addrspace(5)* %Generic; // emits st.local.u32
// :w!
