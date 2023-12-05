import gurobipy as gp

# wrapper for gurobipy model to enable solver-agnostic code.
class GurobiModel(gp.Model):    
    quicksum = gp.quicksum
    minimize = gp.GRB.MINIMIZE

    def addBinaryVar(self, name):
        return self.addVar(vtype=gp.GRB.BINARY, name=name) 
    
    def addContinuousVar(self, name):
        return self.addVar(vtype=gp.GRB.CONTINUOUS, name=name) 

    def setNonconvex(self, value):
        self.params.NonConvex = 2 if value else 0
