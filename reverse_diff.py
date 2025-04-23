import ir
ir.generate_asdl_file()
import _asdl.loma as loma_ir
import irmutator
import autodiff
import string
import random

# From https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_id_generator(size=6, chars=string.ascii_lowercase + string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def reverse_diff(diff_func_id : str,
                 structs : dict[str, loma_ir.Struct],
                 funcs : dict[str, loma_ir.func],
                 diff_structs : dict[str, loma_ir.Struct],
                 func : loma_ir.FunctionDef,
                 func_to_rev : dict[str, str]) -> loma_ir.FunctionDef:
    """ Given a primal loma function func, apply reverse differentiation
        and return a function that computes the total derivative of func.

        For example, given the following function:
        def square(x : In[float]) -> float:
            return x * x
        and let diff_func_id = 'd_square', reverse_diff() should return
        def d_square(x : In[float], _dx : Out[float], _dreturn : float):
            _dx = _dx + _dreturn * x + _dreturn * x

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
        func_to_rev - mapping from primal function ID to its reverse differentiation
    """

    # Some utility functions you can use for your homework.
    def type_to_string(t):
        match t:
            case loma_ir.Int():
                return 'int'
            case loma_ir.Float():
                return 'float'
            case loma_ir.Array():
                return 'array_' + type_to_string(t.t)
            case loma_ir.Struct():
                return t.id
            case _:
                assert False

    def assign_zero(target):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                return [loma_ir.Assign(target, loma_ir.ConstFloat(0.0))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += assign_zero(target_m)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += assign_zero(target_m)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += assign_zero(target_m)
                return stmts
            case _:
                assert False

    def accum_deriv(target, deriv, overwrite):
        match target.t:
            case loma_ir.Int():
                return []
            case loma_ir.Float():
                if overwrite:
                    return [loma_ir.Assign(target, deriv)]
                else:
                    return [loma_ir.Assign(target,
                        loma_ir.BinaryOp(loma_ir.Add(), target, deriv))]
            case loma_ir.Struct():
                s = target.t
                stmts = []
                for m in s.members:
                    target_m = loma_ir.StructAccess(
                        target, m.id, t = m.t)
                    deriv_m = loma_ir.StructAccess(
                        deriv, m.id, t = m.t)
                    if isinstance(m.t, loma_ir.Float):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    elif isinstance(m.t, loma_ir.Int):
                        pass
                    elif isinstance(m.t, loma_ir.Struct):
                        stmts += accum_deriv(target_m, deriv_m, overwrite)
                    else:
                        assert isinstance(m.t, loma_ir.Array)
                        assert m.t.static_size is not None
                        for i in range(m.t.static_size):
                            target_m = loma_ir.ArrayAccess(
                                target_m, loma_ir.ConstInt(i), t = m.t.t)
                            deriv_m = loma_ir.ArrayAccess(
                                deriv_m, loma_ir.ConstInt(i), t = m.t.t)
                            stmts += accum_deriv(target_m, deriv_m, overwrite)
                return stmts
            case _:
                assert False

    def check_lhs_is_output_arg(lhs, output_args):
        match lhs:
            case loma_ir.Var():
                return lhs.id in output_args
            case loma_ir.StructAccess():
                return check_lhs_is_output_arg(lhs.struct, output_args)
            case loma_ir.ArrayAccess():
                return check_lhs_is_output_arg(lhs.array, output_args)
            case _:
                assert False

    # A utility class that you can use for HW3.
    # This mutator normalizes each call expression into
    # f(x0, x1, ...)
    # where x0, x1, ... are all loma_ir.Var or 
    # loma_ir.ArrayAccess or loma_ir.StructAccess
    # Furthermore, it normalizes all Assign statements
    # with a function call
    # z = f(...)
    # into a declaration followed by an assignment
    # _tmp : [z's type]
    # _tmp = f(...)
    # z = _tmp
    class CallNormalizeMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            self.tmp_count = 0
            self.tmp_declare_stmts = []
            new_body = [self.mutate_stmt(stmt) for stmt in node.body]
            new_body = irmutator.flatten(new_body)

            new_body = self.tmp_declare_stmts + new_body

            return loma_ir.FunctionDef(\
                node.id, node.args, new_body, node.is_simd, node.ret_type, lineno = node.lineno)

        def mutate_return(self, node):
            self.tmp_assign_stmts = []
            val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Return(\
                val,
                lineno = node.lineno)]

        def mutate_declare(self, node):
            self.tmp_assign_stmts = []
            val = None
            if node.val is not None:
                val = self.mutate_expr(node.val)
            return self.tmp_assign_stmts + [loma_ir.Declare(\
                node.target,
                node.t,
                val,
                lineno = node.lineno)]

        def mutate_assign(self, node):
            self.tmp_assign_stmts = []
            target = self.mutate_expr(node.target)
            self.has_call_expr = False
            val = self.mutate_expr(node.val)
            if self.has_call_expr:
                # turn the assignment into a declaration plus
                # an assignment
                self.tmp_count += 1
                tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                self.tmp_count += 1
                self.tmp_declare_stmts.append(loma_ir.Declare(\
                    tmp_name,
                    target.t,
                    lineno = node.lineno))
                tmp_var = loma_ir.Var(tmp_name, t = target.t)
                assign_tmp = loma_ir.Assign(\
                    tmp_var,
                    val,
                    lineno = node.lineno)
                assign_target = loma_ir.Assign(\
                    target,
                    tmp_var,
                    lineno = node.lineno)
                return self.tmp_assign_stmts + [assign_tmp, assign_target]
            else:
                return self.tmp_assign_stmts + [loma_ir.Assign(\
                    target,
                    val,
                    lineno = node.lineno)]

        def mutate_call_stmt(self, node):
            self.tmp_assign_stmts = []
            call = self.mutate_expr(node.call)
            return self.tmp_assign_stmts + [loma_ir.CallStmt(\
                call,
                lineno = node.lineno)]

        def mutate_call(self, node):
            self.has_call_expr = True
            new_args = []
            for arg in node.args:
                if not isinstance(arg, loma_ir.Var) and \
                        not isinstance(arg, loma_ir.ArrayAccess) and \
                        not isinstance(arg, loma_ir.StructAccess):
                    arg = self.mutate_expr(arg)
                    tmp_name = f'_call_t_{self.tmp_count}_{random_id_generator()}'
                    self.tmp_count += 1
                    tmp_var = loma_ir.Var(tmp_name, t = arg.t)
                    self.tmp_declare_stmts.append(loma_ir.Declare(\
                        tmp_name, arg.t))
                    self.tmp_assign_stmts.append(loma_ir.Assign(\
                        tmp_var, arg))
                    new_args.append(tmp_var)
                else:
                    new_args.append(arg)
            return loma_ir.Call(node.id, new_args, t = node.t)

    # HW2 happens here. Modify the following IR mutators to perform
    # reverse differentiation.
    class ForwardPassMutator(irmutator.IRMutator):
        def __init__(self, out_args):
            self.var_to_diff_dict = dict()
            self.type_to_stackName_ptrName_stackSize = dict()
            self.out_args = out_args
        
        def mutate_return(self, node):
            print("inside forward mutate_return")
            return []

        def mutate_declare(self, node):
            print("inside forward mutate_declare")
            target = node.target
            dtarget_name = "_d" + target + "_" + random_id_generator()
            new_stmt = loma_ir.Declare(dtarget_name, node.t, lineno=node.lineno)
            self.var_to_diff_dict[target] = dtarget_name
            return [node, new_stmt]
        
        def mutate_assign(self, node):
            print("inside forward mutate_assign")
            target = node.target
            target_type = target.t

            if check_lhs_is_output_arg(target, self.out_args):
                return []

            if target_type not in self.type_to_stackName_ptrName_stackSize:
                rand = random_id_generator()
                stackName = "_t_" + type_to_string(target_type) + "_" + rand
                ptrName = "_stack_ptr_" + type_to_string(target_type) + "_" + rand
                self.type_to_stackName_ptrName_stackSize[target_type] = [stackName, ptrName, 1]    
            else:
                stackName = self.type_to_stackName_ptrName_stackSize[target_type][0]
                ptrName = self.type_to_stackName_ptrName_stackSize[target_type][1]
                self.type_to_stackName_ptrName_stackSize[target_type][2] += 1

            var_stack = loma_ir.Var(stackName)
            var_ptr = loma_ir.Var(ptrName)
            push_to_stack_stmt = loma_ir.Assign(loma_ir.ArrayAccess(var_stack, var_ptr), target)
            advance_ptr_stmt = loma_ir.Assign(var_ptr, loma_ir.BinaryOp(loma_ir.Add(), var_ptr, loma_ir.ConstInt(1)))

            return [push_to_stack_stmt, advance_ptr_stmt, node]


    # Apply the differentiation.
    class RevDiffMutator(irmutator.IRMutator):
        def mutate_function_def(self, node):
            # HW2: TODO
            print("inside mutate_function_def")
            new_args = []
            self.var_to_diff_dict = dict()
            self.is_assign = False
            self.return_input_id = None
            self.assign_adj_count = 0
            self.out_args = []

            for arg in node.args:
                if arg.i == loma_ir.In():
                    new_args.append(arg)
                    new_out_id = "_d" + arg.id + "_" + random_id_generator()
                    new_args.append(loma_ir.Arg(new_out_id, arg.t, loma_ir.Out()))
                    self.var_to_diff_dict[arg.id] = new_out_id
                elif arg.i == loma_ir.Out():
                    new_in_id = "_d" + arg.id + "_" + random_id_generator()
                    new_args.append(loma_ir.Arg(new_in_id, arg.t, loma_ir.In()))
                    self.var_to_diff_dict[arg.id] = new_in_id
                    self.out_args.append(arg.id)
            
            if node.ret_type is not None:
                new_return_input_id = "_dret_" + random_id_generator() 
                new_args.append(loma_ir.Arg(new_return_input_id, node.ret_type, loma_ir.In()))
                self.return_input_id = new_return_input_id

            # forward mode
            forward_mutator = ForwardPassMutator(self.out_args)
            forward_body = irmutator.flatten([forward_mutator.mutate_stmt(stmt) for stmt in node.body])

            self.var_to_diff_dict = self.var_to_diff_dict | forward_mutator.var_to_diff_dict
            print(self.var_to_diff_dict)

            self.type_to_stackName_ptrName_stackSize = forward_mutator.type_to_stackName_ptrName_stackSize
            predeclare_stmt = []
            for t_type in self.type_to_stackName_ptrName_stackSize.keys():
                stackName, ptrName, stackSize = self.type_to_stackName_ptrName_stackSize[t_type]
                predeclare_stmt.append(loma_ir.Declare(stackName, loma_ir.Array(t_type, stackSize)))
                predeclare_stmt.append(loma_ir.Declare(ptrName, loma_ir.Int()))

            print(predeclare_stmt)

            # reverse mode
            reversed_body = irmutator.flatten([self.mutate_stmt(stmt) for stmt in reversed(node.body)])

            # tmp varaiables for mutate_assign
            tmp_declare = []
            for i in range(self.assign_adj_count):
                tmp_declare.append(loma_ir.Declare(f'_adj_{i}', loma_ir.Float()))

            new_body = predeclare_stmt + forward_body + tmp_declare + reversed_body
            new_node = loma_ir.FunctionDef(\
                diff_func_id, 
                new_args, 
                new_body, 
                node.is_simd, 
                None,
                lineno=node.lineno
            )
            return new_node

        def mutate_return(self, node):
            # HW2: TODO
            print("inside mutate_return")
            print(node)
            if self.return_input_id is None:
                return []
            self.adjoint = loma_ir.Var(self.return_input_id, lineno=node.lineno, t=node.val.t)
            return self.mutate_expr(node.val)

        def mutate_declare(self, node):
            # HW2: TODO
            print("inside mutate_declare")
            self.adjoint = loma_ir.Var(self.var_to_diff_dict[node.target], lineno=node.lineno, t=node.t)
            if node.val is None:
                return []
            return self.mutate_expr(node.val)

        def mutate_assign(self, node):
            # HW2: TODO
            print("inside mutate_assign")

            if check_lhs_is_output_arg(node.target, self.out_args):
                self.adjoint = loma_ir.Var(self.var_to_diff_dict[node.target.id], lineno=node.lineno, t=node.target.t)
                return self.mutate_expr(node.val)

            self.is_assign = True
            self.assign_adj_tmp_list = []

            # pop value from stack
            t_type = node.target.t
            self.adjoint = loma_ir.Var(self.var_to_diff_dict[node.target.id], lineno=node.lineno, t=t_type)
            adjoint_copy = self.adjoint
            stackName, ptrName, _ = self.type_to_stackName_ptrName_stackSize[t_type]
            var_stack = loma_ir.Var(stackName)
            var_ptr = loma_ir.Var(ptrName)
            pop_stmt = []
            pop_stmt.append(loma_ir.Assign(var_ptr, loma_ir.BinaryOp(loma_ir.Sub(), var_ptr, loma_ir.ConstInt(1))))
            pop_stmt.append(loma_ir.Assign(node.target, loma_ir.ArrayAccess(var_stack, var_ptr)))

            base_stmt = self.mutate_expr(node.val)

            zero_diff_stmt = loma_ir.Assign(adjoint_copy, loma_ir.ConstFloat(0.0))

            self.is_assign = False
            self.adjoint = adjoint_copy
            return [pop_stmt, self.assign_adj_tmp_list, zero_diff_stmt, base_stmt]

        def mutate_ifelse(self, node):
            # HW3: TODO
            return super().mutate_ifelse(node)

        def mutate_call_stmt(self, node):
            # HW3: TODO
            return super().mutate_call_stmt(node)

        def mutate_while(self, node):
            # HW3: TODO
            return super().mutate_while(node)

        def mutate_const_float(self, node):
            # HW2: TODO
            print("inside mutate_const_float")
            return []

        def mutate_const_int(self, node):
            # HW2: TODO
            print("inside mutate_const_int")
            return []

        def mutate_var(self, node):
            # HW2: TODO
            print("inside mutate_var")
            if self.is_assign:
                adj_name = f'_adj_{self.assign_adj_count}'
                self.assign_adj_count += 1
                var_adj = loma_ir.Var(adj_name, t=loma_ir.Float())
                self.assign_adj_tmp_list.append(accum_deriv(var_adj, self.adjoint, True))

                dvar = loma_ir.Var(self.var_to_diff_dict[node.id], lineno=node.lineno, t=node.t)
                return [accum_deriv(dvar, var_adj, False)]
            else:
                dvar = loma_ir.Var(self.var_to_diff_dict[node.id], lineno=node.lineno, t=node.t)
                return [accum_deriv(dvar, self.adjoint, False)]

        def mutate_array_access(self, node):
            # HW2: TODO
            return super().mutate_array_access(node)

        def mutate_struct_access(self, node):
            # HW2: TODO
            return super().mutate_struct_access(node)

        def mutate_add(self, node):
            # HW2: TODO
            print("inside mutate_add")
            left_stmt_list = self.mutate_expr(node.left)
            right_stmt_list = self.mutate_expr(node.right)
            return left_stmt_list + right_stmt_list

        def mutate_sub(self, node):
            # HW2: TODO
            print("inside mutate_sub")
            left_stmt_list = self.mutate_expr(node.left)
            adjoint_copy = self.adjoint
            self.adjoint = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), adjoint_copy)
            right_stmt_list = self.mutate_expr(node.right)
            self.ajoint = adjoint_copy
            return left_stmt_list + right_stmt_list

        def mutate_mul(self, node):
            # HW2: TODO
            print("inside mutate_mul")
            adjoint_copy = self.adjoint

            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, node.right)
            left_stmt_list = self.mutate_expr(node.left)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, node.left)
            right_stmt_list = self.mutate_expr(node.right)
            self.adjoint = adjoint_copy
            return left_stmt_list + right_stmt_list

        def mutate_div(self, node):
            # HW2: TODO
            print("inside mutate_div")
            adjoint_copy = self.adjoint

            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), adjoint_copy, node.right)
            left_stmt_list = self.mutate_expr(node.left)
            neg_adj = loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), adjoint_copy)
            neg_adj_times_x = loma_ir.BinaryOp(loma_ir.Mul(), neg_adj, node.left)
            y_sqred = loma_ir.BinaryOp(loma_ir.Mul(), node.right, node.right)
            self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), neg_adj_times_x, y_sqred)
            right_stmt_list = self.mutate_expr(node.right)
            self.adjoint = adjoint_copy
            return left_stmt_list + right_stmt_list

        def mutate_call(self, node):
            # HW2: TODO
            print("inside mutate_call")

            match node.id:
                case "sin":
                    assert(len(node.args) == 1)
                    adjoint_copy = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, loma_ir.Call("cos", node.args))
                    stmt = [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = adjoint_copy
                    return stmt

                case "cos":
                    assert(len(node.args) == 1)
                    adjoint_copy = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, loma_ir.BinaryOp(loma_ir.Sub(), loma_ir.ConstFloat(0.0), loma_ir.Call("sin", node.args)))
                    stmt = [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = adjoint_copy
                    return stmt

                case "sqrt":
                    assert(len(node.args) == 1)
                    adjoint_copy = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), adjoint_copy, loma_ir.BinaryOp(loma_ir.Mul(), loma_ir.ConstFloat(2.0), node))
                    stmt = [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = adjoint_copy
                    return stmt

                case "pow":
                    assert(len(node.args) == 2)
                    adjoint_copy = self.adjoint
                    x = node.args[0]
                    y = node.args[1]
                    x_to_ySUBone = loma_ir.Call("pow", [x, loma_ir.BinaryOp(loma_ir.Sub(), y, loma_ir.ConstFloat(1.0))])
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, loma_ir.BinaryOp(loma_ir.Mul(), y, x_to_ySUBone))
                    left_stmt = self.mutate_expr(x)
                    log_x = loma_ir.Call("log", [x])
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, loma_ir.BinaryOp(loma_ir.Mul(), node, log_x))
                    right_stmt = self.mutate_expr(y)
                    self.adjoint = adjoint_copy
                    return left_stmt + right_stmt
                
                case "exp":
                    assert(len(node.args) == 1)
                    adjoint_copy = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Mul(), adjoint_copy, node)
                    stmt = [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = adjoint_copy
                    return stmt
                
                case "log":
                    assert(len(node.args) == 1)
                    adjoint_copy = self.adjoint
                    self.adjoint = loma_ir.BinaryOp(loma_ir.Div(), adjoint_copy, node.args[0])
                    stmt = [self.mutate_expr(arg) for arg in node.args]
                    self.adjoint = adjoint_copy
                    return stmt
                
                case "int2float":
                    assert(len(node.args) == 1)
                    return []

                case "float2int":
                    assert(len(node.args) == 1)
                    return []

                case _:
                    assert False
                    return super().mutate_call(node)

    return RevDiffMutator().mutate_function_def(func)
