import gurobipy as gp


# wrapper for gurobipy model to enable solver-agnostic code.
class GurobiModel(gp.Model):    
    quicksum = gp.quicksum
    minimize = gp.GRB.MINIMIZE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.params.NonConvex = 0

    def setLogFile(self, filename):
        self.setParam(gp.GRB.Param.LogFile, filename)

    def addBinaryVar(self, name):
        return self.addVar(vtype=gp.GRB.BINARY, name=name) 
    
    def addContinuousVar(self, name, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY):
        return self.addVar(vtype=gp.GRB.CONTINUOUS, name=name, lb=lb, ub=ub)

    def setNonconvex(self, value):
        self.params.NonConvex = 2 if value else 0

    def setLazy(self, constraint, value):
        constraint.setAttr(gp.GRB.Attr.Lazy, value)

    def getVal(self, var):
        return var.X
