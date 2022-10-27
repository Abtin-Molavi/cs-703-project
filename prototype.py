from pysat.formula import IDPool


def initial_matrix_is_identity(num_qubits, vpool):
    clauses = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                clauses.append(vpool.id(("a", 0, i, j)))
            else: 
                clauses.append(-vpool.id(("a", 0, i, j)))
    return clauses

def final_matrix_properties(num_qubits, vpool):
    clauses = []
    for 