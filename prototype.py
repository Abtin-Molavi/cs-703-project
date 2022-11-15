from pysat.formula import IDPool
from pysat.card import CardEnc
from pysat.solvers import Solver
from qiskit.circuit import QuantumCircuit
import math

def initial_matrix_is_identity(num_qubits, vpool):
    clauses = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i == j:
                clauses.append([vpool.id(("a", i, j, 0))])
            else: 
                clauses.append([-vpool.id(("a", i, j, 0))])
    return clauses

def final_matrix_properties(num_qubits, vpool, g_mat, fs, k):
    clauses = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            if g_mat[i][j] == 1:            
                clauses.append([vpool.id(("a", i, j, k))])
            else:
                clauses.append([-vpool.id(("a", i, j, k))])
    
    for f in range(len(fs)):
        clauses.append([vpool.id(('p', i, lk, f)) for i in range(num_qubits) for lk in range(k+1)])
        for i in range(num_qubits):
            for lk in range(k+1):
                for j in range(num_qubits):
                    f_value =  fs[f][j]
                    if f_value == 1:
                        j_term = vpool.id(('a', i , j, lk))
                    else: 
                        j_term = -vpool.id(('a', i, j, lk))
                    clauses.append([-vpool.id(('p', i, lk, f)), j_term])
    return clauses

def cnot_well_defined(num_qubits,vpool, k, aux_vars):
    clauses = []
    for lk in range(k):
        #print(k)
        #print([[-vpool.id(('q', i, lk)), -vpool.id(('t', i , lk)) ] for i in range(num_qubits)])
        clauses.extend([[-vpool.id(('q', i, lk)), -vpool.id(('t', i , lk)) ] for i in range(num_qubits)])
        clauses.extend(CardEnc.equals([vpool.id(('q', i, lk)) for i in range(num_qubits)], 1, vpool=aux_vars).clauses)
        clauses.extend(CardEnc.equals([vpool.id(('t', i, lk)) for i in range(num_qubits)], 1, vpool=aux_vars).clauses)


    return clauses

def h_encodes_activated(num_qubits, vpool, k,):
    clauses = []
    for i in range(num_qubits):
        for j in range(num_qubits):
            for lk in range(k):
                clauses.append([-vpool.id(('h', i, j, lk)), vpool.id(('q', i, lk))] )
                clauses.append([-vpool.id(('h', i, j, lk)), vpool.id(('a', i, j, lk))])
                clauses.append([vpool.id(('h', i, j, lk)), -vpool.id(('a', i, j, lk)), -vpool.id(('q', i, lk))])
    return clauses


def transformation_clauses(num_qubits, vpool, k,):
    clauses = []
    for lk in range(k):
            for j in range(num_qubits):
                for l in range(num_qubits):
                    clauses.append([-vpool.id(('a', j, l, lk+1)), vpool.id(('a', j, l, lk)), vpool.id(('t', j, lk))])
                    clauses.append([-vpool.id(('a', j, l, lk+1)), vpool.id(('a', j, l, lk)),] + [vpool.id(('h', i, l, lk)) for i in range(num_qubits)])
                    clauses.append([vpool.id(('a', j, l, lk+1)), -vpool.id(('a', j, l, lk)), vpool.id(('t', j, lk))])
                    clauses.append([vpool.id(('a', j, l, lk+1)), -vpool.id(('a', j, l, lk)),] + [vpool.id(('h', i, l, lk)) for i in range(num_qubits)])
                    for i in range(num_qubits):
                        clauses.append([-vpool.id(('a', j, l, lk+1)), -vpool.id(('a', j, l, lk)),  -vpool.id(('t', j, lk)), -vpool.id(('h', i, l, lk))])   
                        clauses.append([vpool.id(('a', j, l, lk+1)), vpool.id(('a', j, l, lk)),  -vpool.id(('t', j, lk)), -vpool.id(('h', i, l, lk))])        
    return clauses
                    # rhs_xor_neg = [-vpool.id(('a', j, l, lk)), -vpool.id(('a', j, l, lk-1)), -vpool.id(('t', j, lk)), -vpool.id(('q', i, lk)), -vpool.id(('a', i,j,lk-1)) ]
                    # clause = [-vpool.id(('a', j, l, lk)), ] + rhs_xor_neg
                    # rhs_xor_pos1 =  [-vpool.id(('a', j, l, lk)), vpool.id(('a', j, l, lk-1)), vpool.id(('t', j, lk))]
                    # rhs_xor_pos2 = [-vpool.id(('a', j, l, lk)), vpool.id(('a', j, l, lk-1)), vpool.id(('q', i, lk))]
                    # rhs_xor_pos3 =  [-vpool.id(('a', j, l, lk)), vpool.id(('a', j, l, lk-1)), vpool.id(('a', i,j,lk-1))]
                     
                     
                    #  -vpool.id(('q', i, lk)), -vpool.id(('a', i,j,lk-1)) ]
def pretty_print_var(x, vpool, aux_vars):
        if vpool.obj(abs(x)):
            return(vpool.obj(x) if x > 0 else ("-",) + vpool.obj(-x))
        else: 
            return("aux_var")

def solve_k(num_qubits,  g_mat, fs, k):
    num_a  = num_qubits*num_qubits*(k+1)
    num_q = num_qubits*k
    num_t = num_q
    num_p = num_qubits*len(fs)*(k+1)
    num_h = num_qubits*num_qubits*k
    top = num_a+num_p+num_q+num_t+num_h
    vpool = IDPool()
    aux_vars = IDPool(start_from=top+1)
    clauses = (
            initial_matrix_is_identity(num_qubits, vpool) +
            final_matrix_properties(num_qubits, vpool, g_mat, fs, k) +
            cnot_well_defined(num_qubits,vpool, k, aux_vars) +
            h_encodes_activated(num_qubits, vpool, k,) +
            transformation_clauses(num_qubits, vpool, k,)
            ) 
    # for clause in cnot_well_defined(num_qubits,vpool, k, aux_vars):
    #     print([vpool.obj(x) for x in clause])        
    with Solver(name='cd', ) as s:
        for clause in clauses:
            s.add_clause(clause)
            # print([pretty_print_var(x, vpool, aux_vars) for x in clause])
            # print(s.solve())
            s.solve()
        if s.get_status():
            # print(s.get_model())
            # print([vpool.obj(x)  for x in s.get_model() if x > 0])
            return s.get_status(),[vpool.obj(x) for x in s.get_model() if x > 0]
        return s.get_status(),[]

def solve(num_qubits,  g_mat, fs):
    solved = False
    k = 0
    while not solved:
        k += 1
        solved, model = solve_k(num_qubits,  g_mat, fs, k)
    return k, model

def extract_circuit(num_qubits, k, cs, model):
    circuit = QuantumCircuit(num_qubits)
    applied_rz = [False] * len(cs)
    for lk in range(k+1):
        c = None
        t = None
        for v in model:
            if v is None: continue
            if v[2] == lk:
                if v[0] == "q":
                    c = v[1]
                if v[0] == "t":
                    t = v[1]
                if v[0] == "p" and not applied_rz[v[3]]:
                    circuit.rz(cs[v[3]], v[1])
                    applied_rz[v[3]] = True
        if lk < k:
            circuit.cx(c, t)
    return circuit


if __name__ == "__main__":
    num_qubits = 3
    g_mat = [
        [1, 0, 0], 
        [1, 1, 0], 
        [1, 1, 1], 
        ]
    fs = [[1, 1, 0], [1, 1, 1]]
    cs = [math.pi/4, 7*math.pi/4]
    k, model = solve(num_qubits, g_mat, fs)
    print(k)
    print(model)
    print(extract_circuit(num_qubits, k, cs, model).qasm())