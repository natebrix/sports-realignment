import pyscipopt as scip

# wrapper for scip model to enable solver-agnostic code.
# pysciopt's API is pretty much a clone of Gurobi's.
class ScipModel(scip.Model):    
    def __init__(self):
        self.nonconvex = False

    def addBinaryVar(self, name):
        return self.addVar(vtype="B", name=name) 
    
    quicksum = scip.quicksum
    minimize = "minimize"

    def setNonconvex(self):
        self.nonconvex = True

    def addConstr(self, c):
        self.addCons(c)

    def setObjective(self, expr, sense):
        if self.nonconvex:
            # Nonconvex 
            self.cost = self.addVar(vtype="C", name="__cost")
            super().setObjective(self.cost, sense)
            self.addCons(self.cost == expr)
        else:
            super().setObjective(expr, sense)

