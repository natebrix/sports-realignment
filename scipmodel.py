import pyscipopt as scip

# wrapper for scip model to enable solver-agnostic code.
# pysciopt's API is pretty much a clone of Gurobi's.
class ScipModel(scip.Model):    
    def __init__(self):
        self.nonconvex = False

    def addBinaryVar(self, name):
        return self.addVar(vtype="B", name=name) 

    def addContinuousVar(self, name):
        return self.addVar(vtype="C", name=name) 

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

