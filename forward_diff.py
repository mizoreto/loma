import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff

def forward_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_fwd : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply forward differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', forward_diff() should return
        def d_square(x : In[_dfloat]) -> _dfloat:
            return make__dfloat(x.val * x.val, x.val * x.dval + x.dval * x.val)
        where the class _dfloat is
        class _dfloat:
            val : float
            dval : float
        and the function make__dfloat is
        def make__dfloat(val : In[float], dval : In[float]) -> _dfloat:
            ret : _dfloat
            ret.val = val
            ret.dval = dval
            return ret

        Parameters:
        diff_func_id - the ID of the returned function
        structs - a dictionary that maps the ID of a Struct to 
                the corresponding Struct
        funcs - a dictionary that maps the ID of a function to 
                the corresponding func
        diff_structs - a dictionary that maps the ID of the primal
                Struct to the corresponding differential Struct
                e.g., diff_structs['float'] returns _dfloat
        func - the function to be differentiated
        func_to_fwd - mapping from primal function ID to its forward differentiation
    """

    # HW1 happens here. Modify the following IR mutators to perform
    # forward differentiation.

    # Apply the differentiation.
    class FwdDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW1: TODO
            print("inside mutate_function_def")

            new_args = [loma_ir.Arg(
                arg.id, 
                autodiff.type_to_diff_type(diff_structs, arg.t),
                arg.i) for arg in node.args]
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_node = loma_ir.FunctionDef(\
                diff_func_id, 
                new_args, 
                new_body, 
                node.is_simd, 
                autodiff.type_to_diff_type(diff_structs, node.ret_type),
                lineno = node.lineno
            )
            return new_node

        def mutate_return(self, node):
            # HW1: TODO
            print("inside mutate_return")
            
            val, dval = self.mutate_expr(node.val)
            if func.ret_type == loma_ir.Float():
                return loma_ir.Return(loma_ir.Call('make__dfloat', [val, dval]), lineno=node.lineno)

            return loma_ir.Return(val, lineno=node.lineno)

        def mutate_declare(self, node):
            # HW1: TODO
            print("inside mutate_declare")

            new_val = node.val

            if node.val is not None:
                val, dval = self.mutate_expr(node.val)
                if isinstance(val.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct)):
                    new_val = val
                else:
                    new_val = loma_ir.Call('make__dfloat', [val, dval])

            return loma_ir.Declare(
                node.target,
                autodiff.type_to_diff_type(diff_structs, node.t),
                new_val,
                lineno=node.lineno)

        def mutate_assign(self, node):
            # HW1: TODO
            print("inside mutate_assign")

            new_target = node.target
            if isinstance(new_target, loma_ir.ArrayAccess):
                new_array = new_target.array
                new_index = self.mutate_expr(new_target.index)[0]
                new_target = loma_ir.ArrayAccess(new_array, new_index)

            val, dval = self.mutate_expr(node.val)

            if isinstance(val.t, (loma_ir.Int, loma_ir.Array, loma_ir.Struct)) or (isinstance(val, loma_ir.Call) and val.id == "float2int"):
                return loma_ir.Assign(node.target, val, lineno=node.lineno)

            return loma_ir.Assign(
                new_target,
                loma_ir.Call('make__dfloat', [val, dval], lineno=node.lineno)
            )

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW1: TODO
            print("inside mutate_const_float")

            return loma_ir.ConstFloat(node.val), loma_ir.ConstFloat(0.0)

        def mutate_const_int(self, node):
            # HW1: TODO
            print("inside mutate_const_int")
            return node, loma_ir.ConstFloat(0.0)

        def mutate_var(self, node):
            # HW1: TODO
            print("inside mutate_var")

            if node.t == loma_ir.Float():
                val = loma_ir.StructAccess(node, 'val')
                dval = loma_ir .StructAccess(node, 'dval')
                return val, dval

            return node, loma_ir.ConstFloat(0.0)

        def mutate_array_access(self, node):
            # HW1: TODO
            print("inside mutate_array_access")

            new_array = node.array
            new_index = self.mutate_expr(node.index)[0]

            if new_array.t.t == loma_ir.Float():
                val = loma_ir.StructAccess(loma_ir.ArrayAccess(new_array, new_index), 'val')
                dval = loma_ir.StructAccess(loma_ir.ArrayAccess(new_array, new_index), 'dval')
                return val, dval
                
            return loma_ir.ArrayAccess(new_array, new_index, lineno=node.lineno, t=node.t), loma_ir.ConstFloat(0.0)

        def mutate_struct_access(self, node):
            # HW1: TODO
            print("inside mutate_struct_access")
            
            if node.t == loma_ir.Float():
                val = loma_ir.StructAccess(node, 'val')
                dval = loma_ir.StructAccess(node, 'dval')
                return val, dval
            
            return node, loma_ir.ConstFloat(0.0)

        def mutate_add(self, node):
            # HW1: TODO
            print("inside mutate_add")

            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            return loma_ir.BinaryOp(loma_ir.Add(), left_val, right_val, lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Add(), left_dval, right_dval, lineno=node.lineno, t=node.t)

        def mutate_sub(self, node):
            # HW1: TODO
            print("inside mutate_sub")

            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            return loma_ir.BinaryOp(loma_ir.Sub(), left_val, right_val, lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Sub(), left_dval, right_dval, lineno=node.lineno, t=node.t)

        def mutate_mul(self, node):
            # HW1: TODO
            print("inside mutate_mul")

            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            xdy = loma_ir.BinaryOp(loma_ir.Mul(), left_val, right_dval)     # x * dy
            ydx = loma_ir.BinaryOp(loma_ir.Mul(), right_val, left_dval)     # y * dx

            return loma_ir.BinaryOp(loma_ir.Mul(), left_val, right_val, lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Add(), xdy, ydx, lineno=node.lineno, t=node.t)

        def mutate_div(self, node):
            # HW1: TODO
            print("inside mutate_div")

            left_val, left_dval = self.mutate_expr(node.left)
            right_val, right_dval = self.mutate_expr(node.right)

            xdy = loma_ir.BinaryOp(loma_ir.Mul(), left_val, right_dval)     # x * dy
            ydx = loma_ir.BinaryOp(loma_ir.Mul(), right_val, left_dval)     # y * dx
            y2 = loma_ir.BinaryOp(loma_ir.Mul(), right_val, right_val)      # y^2

            ydxSUBxdy = loma_ir.BinaryOp(loma_ir.Sub(), ydx, xdy)           # y*dx - x*dy

            return loma_ir.BinaryOp(loma_ir.Div(), left_val, right_val, lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Div(), ydxSUBxdy, y2, lineno=node.lineno, t=node.t)

        def mutate_call(self, node):
            # HW1: TODO
            print("inside mutate_call")

            match node.id:
                case "sin":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    return loma_ir.Call("sin", [val], lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Mul(), loma_ir.Call("cos", [val]), dval, lineno=node.lineno, t=node.t)
                case "cos":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    sin_x = loma_ir.Call("sin", [val])
                    sinx_dx = loma_ir.BinaryOp(loma_ir.Mul(), sin_x, dval)
                    return loma_ir.Call("cos", [val], lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Mul(), loma_ir.ConstInt(-1), sinx_dx, lineno=node.lineno, t=node.t)
                case "sqrt":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    sqrtx = loma_ir.Call("sqrt", [val], lineno=node.lineno, t=node.t)
                    return sqrtx, loma_ir.BinaryOp(loma_ir.Div(), dval, loma_ir.BinaryOp(loma_ir.Mul(), loma_ir.ConstInt(2), sqrtx), lineno=node.lineno, t=node.t)
                case "pow":
                    assert(len(node.args) == 2)
                    base_val, base_dval = self.mutate_expr(node.args[0])
                    exp_val, exp_dval = self.mutate_expr(node.args[1])
                    x_to_y = loma_ir.Call("pow", [base_val, exp_val], lineno=node.lineno, t=node.t)
                    x_to_ySUBone = loma_ir.Call("pow", [base_val, loma_ir.BinaryOp(loma_ir.Sub(), exp_val, loma_ir.ConstInt(1))])
                    logx = loma_ir.Call("log", [base_val])
                    temp_left = loma_ir.BinaryOp(loma_ir.Mul(), exp_val, x_to_ySUBone)
                    temp_right = loma_ir.BinaryOp(loma_ir.Mul(), x_to_y, logx)
                    dx_temp_left = loma_ir.BinaryOp(loma_ir.Mul(), base_dval, temp_left)
                    dy_temp_right = loma_ir.BinaryOp(loma_ir.Mul(), exp_dval, temp_right)

                    return x_to_y, loma_ir.BinaryOp(loma_ir.Add(), dx_temp_left, dy_temp_right, lineno=node.lineno, t=node.t)
                case "exp":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    exp_x = loma_ir.Call("exp", [val], lineno=node.lineno, t=node.t)

                    return exp_x, loma_ir.BinaryOp(loma_ir.Mul(), exp_x, dval, lineno=node.lineno, t=node.t)
                case "log":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    return loma_ir.Call("log", [val], lineno=node.lineno, t=node.t), loma_ir.BinaryOp(loma_ir.Div(), dval, val, lineno=node.lineno, t=node.t)
                case "int2float":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    return loma_ir.Call("int2float", [val], lineno=node.lineno, t=node.t), loma_ir.ConstFloat(0.0)
                case "float2int":
                    assert(len(node.args) == 1)
                    val, dval = self.mutate_expr(node.args[0])
                    return loma_ir.Call("float2int", [val], lineno=node.lineno, t=node.t), loma_ir.ConstInt(0)
                case _ :
                    return super().mutate_struct_access(node)


    return FwdDiffMutator().mutate_function_def(func)
