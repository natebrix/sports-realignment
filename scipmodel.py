import pyscipopt as scip

# wrapper for scip model to enable solver-agnostic code.
# pysciopt's API is pretty much a clone of Gurobi's.
class ScipModel(scip.Model):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.nonconvex = False

    def addBinaryVar(self, name):
        return self.addVar(vtype="B", name=name) 

    def addContinuousVar(self, name, lb=float('-inf'), ub=float('inf')):
        return self.addVar(vtype="C", name=name, lb=lb, ub=ub) 

    def setLogFile(self, filename):
        #self.setParam(gp.GRB.Param.LogFile, filename)
        pass # todo

    quicksum = scip.quicksum
    minimize = "minimize"

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
        else:
            super().setObjective(expr, sense)

    def setLazy(self, constraint, value):
        pass
        #constraint.setAttr(gp.GRB.Attr.Lazy, value)

    def getVal(self, var):
        return super().getVal(var)

    def write(self, filename):
        self.writeProblem(filename)

