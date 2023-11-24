import pyscipopt as scip

# wrapper for scip model to enable solver-agnostic code.
# pysciopt's API is pretty much a clone of Gurobi's.
class ScipModel(scip.Model):    
    def addBinaryVar(self, name):
        return self.addVar(vtype="B", name=name) 
    
    quicksum = scip.quicksum
    minimize = "minimize"

    def addConstr(self, c):
        self.addCons(c)

