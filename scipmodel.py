import pyscipopt as scip

class ScipConshdlr(scip.Conshdlr):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def conslock(self, constraint, locktype, nlockspos, nlocksneg):
         pass
    
    # Method to check constraints
    def conscheck(self, constraints, solution, check_integrality,
                  check_lp_rows, print_reason, completely):
        violated = self.callback(self) # todo the model??  
        return {'result': scip.SCIP_RESULT.INFEASIBLE if violated else scip.SCIP_RESULT.FEASIBLE}
    
# wrapper for scip model to enable solver-agnostic code.
# pysciopt's API is pretty much a clone of Gurobi's.
class ScipModel(scip.Model):   
    solver_name = 'scip'
    quicksum = scip.quicksum
    minimize = "minimize"
    maximize = "maximize"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.nonconvex = False
        self.nonconvex_expr = None

    def isNonconvex(self):
        return self.nonconvex

    def addBinaryVar(self, name):
        return self.addVar(vtype="B", name=name) 
    
    def addIntegerVar(self, name, lb=float('-inf'), ub=float('inf')):
        return self.addVar(vtype="I", name=name, lb=lb, ub=ub)

    def addContinuousVar(self, name, lb=float('-inf'), ub=float('inf')):
        return self.addVar(vtype="C", name=name, lb=lb, ub=ub) 

    def setLogFile(self, filename):
        #self.setParam(gp.GRB.Param.LogFile, filename)
        pass # todo

    def setNonconvex(self, value):
        self.nonconvex = value

    def addConstr(self, c):
        self.addCons(c)

    def setObjective(self, expr, sense):
        if self.nonconvex:
            # SCIP does not directly support a nonlinear objective, but a constraint is a workaround:
            #
            # min <expr>    ; <expr> is nonlinear
            # becomes
            #
            # min cost
            # s.t. cost == <expr>
            self.cost = self.addVar(vtype="C", name="__cost")
            super().setObjective(self.cost, sense)
            self.addCons(self.cost == expr)
            self.nonconvex_expr = expr
        else:
            super().setObjective(expr, sense)

    def setLazy(self, constraint, value):
        pass
        #constraint.setAttr(gp.GRB.Attr.Lazy, value)

    def getVal(self, var):
        return super().getVal(var)

    def write(self, filename):
        self.writeProblem(filename)

    def update(self):
        pass

    def warm(self, s, v): # todo rename me
        return self.getSolVal(s, v) > 0.99

    def both_warm(self, s, vs):  # todo rename me
        return False if len(vs) != 2 else self.warm(s, vs[0]) and self.warm(s, vs[1])

    def setNonconvexSolVal(self, s):
        if self.nonconvex:
            # these are the terms in the dummy expression we added. Let's fish out the variables and get their values.
            ts = self.nonconvex_expr.terms
            cost_val = sum([ts[xt] for xt in ts.keys() if self.both_warm(s, xt.vartuple)])
            self.setSolVal(s, self.cost, cost_val)

    def is_optimal(self):
        return self.getStatus() == "optimal"

    def registerConstraintCallback(self, callback):
        conshdlr = ScipConshdlr(callback)
        self.includeConshdlr(conshdlr, "ScipConshdlr", "Custom constraint handler", enfopriority = 0,
                             needscons=False)

