import cvxpy as cp
import numpy as np

'''
Solves the following 4x4 sudoku puzzle:
[[2 , 1 , - , - ],
 [- , 3 , 2 , - ],
 [- , - , - , 4 ],
 [1 , - , - , - ]
'''

vars = []
for i in range(4):
    vars.append([])
    for j in range(4):
        vars[i].append([])
        for k in range(4):
            vars[i][j].append(cp.Variable(boolean=True))

constraints = []
for r in range(4):
    for c in range(4):
        s = 0
        for v in range(4):
            s += vars[r][c][v]
        constraints.append(s == 1)

for v in range(4):
    for c in range(4):
        s = 0
        for r in range(4):
            s += vars[r][c][v]
        constraints.append(s == 1)

for v in range(4):
    for r in range(4):
        s = 0
        for c in range(4):
            s += vars[r][c][v]
        constraints.append(s == 1)

for p1 in range(2):
    for p2 in range(2):
        for r in range(2*p1, 2*p1 + 2):
            for c in range(2*p2, 2*p2 + 2):
                s = 0
                for v in range(4):
                    s += vars[r][c][v]
                constraints.append(s == 1)

constraints.append(vars[0][0][1] == 1)
constraints.append(vars[0][1][0] == 1)
constraints.append(vars[1][1][2] == 1)
constraints.append(vars[3][0][0] == 1)
constraints.append(vars[1][2][1] == 1)
constraints.append(vars[2][3][3] == 1)
prob = cp.Problem(cp.Minimize(0), constraints)
prob.solve(solver='ECOS_BB')

solution =  np.zeros((4, 4))
print(vars[0][0][1].value)
for r in range(4):
    for c in range(4):
        for v in range(4):
            if vars[r][c][v].value > 0.9:
                solution[r][c] = v + 1
print(solution)
