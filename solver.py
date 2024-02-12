class SolverRegistry(): 
    def __init__(self, solvers=None):
        self.solvers = {}
        for s in solvers:
            self.register_solver(s)

    def register_solver(self, solver):
        self.solvers[solver.solver_name] = solver

    def create_solver_environment(self, solver):
        if solver in self.solvers:
            return self.solvers[solver]
        else:
            raise Exception(f'Unknown solver {solver}. Registered solvers: {", ".join(self.solvers.keys())}')
