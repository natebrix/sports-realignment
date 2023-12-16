import gurobipy as gp

# wrapper for gurobipy model to enable solver-agnostic code.
class GurobiModel(gp.Model):    
    quicksum = gp.quicksum
    minimize = gp.GRB.MINIMIZE

    def addBinaryVar(self, name):
        return self.addVar(vtype=gp.GRB.BINARY, name=name) 
    
    def addContinuousVar(self, name, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY):
        return self.addVar(vtype=gp.GRB.CONTINUOUS, name=name, lb=lb, ub=ub)

    def setNonconvex(self, value):
        self.params.NonConvex = 2 if value else 0
