from substitution_cipher import GeneticSubstitutionSolver, SolverType
from pathlib import Path
import sys

ciphertext = Path(sys.argv[1]).read_text()
solver_type = "regular"
verbose_display = False

if len(sys.argv) > 2:
    solver_type = sys.argv[2]

if len(sys.argv) > 3:
    verbose_display = sys.argv[3]


# Check if the provided solver type is valid
if solver_type not in ['regular', 'lamark', 'darwin']:
    print("Invalid solver type. Please choose one of: 'regular', 'lamark', 'darwin'")
    sys.exit(1)

solver = GeneticSubstitutionSolver(ciphertext)

plaintext = ''

if solver_type == 'regular':
    plaintext = solver.solve(SolverType.REGULAR)
elif solver_type == 'lamark':
    plaintext = solver.solve(SolverType.LAMARK)
elif solver_type == 'darwin':
    plaintext = solver.solve(SolverType.DARWIN)

print(plaintext)