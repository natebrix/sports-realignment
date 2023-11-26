import gurobipy as gp

# wrapper for gurobipy model to enable solver-agnostic code.
class GurobiModel(gp.Model):    
    def addBinaryVar(self, name):
        return self.addVar(vtype=gp.GRB.BINARY, name=name) 
    
    quicksum = gp.quicksum
    minimize = gp.GRB.MINIMIZE

    def setNonconvex(self):
        self.params.NonConvex = 2 
