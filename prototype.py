import os
from pysat.formula import IDPool
from pysat.card import CardEnc
from pysat.solvers import Solver
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import SabreLayout, SabreSwap, ApplyLayout
from qiskit.transpiler import PassManager, CouplingMap
import math
from architectures import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

def injective_initial_map(num_qubits, vpool, cm, aux_vars):
    clauses = []
    for j in range(len(cm)):
        clauses.extend(CardEnc.atmost([vpool.id(('m', i, j)) for i in range(num_qubits)], 1, vpool=aux_vars).clauses)
    for i in range(num_qubits):
        clauses.extend(CardEnc.equals([vpool.id(('m', i, j)) for j in range(len(cm))], 1, vpool=aux_vars).clauses)
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

def cnots_executable(num_qubits ,vpool, k, cm):
    clauses = []
    for lk in range(k):
        for i in range(num_qubits):
            for j in range(num_qubits):
                for p1 in range(len(cm)):
                    adjacent = [p2 for [p2] in np.argwhere(cm[p1] > 0)]
                    clauses.append([-vpool.id(('q', i, lk)), -vpool.id(("t", j, lk)), -vpool.id(('m', i, p1)) ] + [vpool.id(("m", j, p2)) for p2 in adjacent])
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

def solve_k(num_qubits,  g_mat, fs, k, cm=None):
    num_a  = num_qubits*num_qubits*(k+1)
    num_q = num_qubits*k
    num_t = num_q
    num_p = num_qubits*len(fs)*(k+1)
    num_h = num_qubits*num_qubits*k
    num_m = num_qubits*len(cm) if cm is not None else 0
    top = num_a+num_p+num_q+num_t+num_h+num_m
    vpool = IDPool()
    aux_vars = IDPool(start_from=top+1)
    clauses = (
            initial_matrix_is_identity(num_qubits, vpool) +
            final_matrix_properties(num_qubits, vpool, g_mat, fs, k) +
            cnot_well_defined(num_qubits,vpool, k, aux_vars) +
            h_encodes_activated(num_qubits, vpool, k,) +
            transformation_clauses(num_qubits, vpool, k,) +
            ([] if cm is None else cnots_executable(num_qubits ,vpool, k, cm) + injective_initial_map(num_qubits, vpool, cm, aux_vars) )
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

def solve(num_qubits,  g_mat, fs, cm=None):
    solved = False
    k = 0
    while not solved:
        print(k)
        solved, model = solve_k(num_qubits,  g_mat, fs, k, cm)
        k += 1
    return k-1, model

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

def layout_circuit(file_name, coupling_map, route=False):
    circ = QuantumCircuit.from_qasm_file(file_name)
    cm = CouplingMap(np.argwhere(coupling_map > 0))
    if route:
        pm = PassManager([SabreLayout(coupling_map=cm, routing_pass=SabreSwap(cm, heuristic="lookahead")), SabreSwap(coupling_map=cm, heuristic="lookahead")]) 
        pm.run(circ).qasm(filename="mapped_and_routed_"+os.path.basename(file_name))
    else:
        pm = PassManager([SabreLayout(coupling_map=cm, routing_pass=SabreSwap(cm, heuristic="lookahead"))]) 
        pm.run(circ).qasm(filename="mapped_"+os.path.basename(file_name))
    return os.path.basename(file_name)

def extract_G_fs(circuit):
    state = np.identity(circuit.num_qubits, dtype=int).tolist()
    fs = []
    for g in circuit.data:
        if g.operation.name == "cx":
            c = g.qubits[0].index
            t = g.qubits[1].index
            state[t] = [1 if state[c][i] != state[t][i] else 0 for i in range(circuit.num_qubits)]
        elif g.operation.name == "rz":
            angle = g.operation.params[0]
            q = g.qubits[0].index
            fs.append((state[q], angle))
        else:
            raise RuntimeError(f"unexpected gate: {g.operation.name}")
    return state, fs

def full_run(og_circuit_filename, arch):
    mapped_file_name = layout_circuit(og_circuit_filename, arch)
    mapped_circuit = QuantumCircuit.from_qasm_file(og_circuit_filename)

    G, fs = extract_G_fs(mapped_circuit)
    fs, cs = zip(*fs)
    k, model = solve(mapped_circuit.num_qubits, G, fs, arch)

    # print(k)
    # print(model)
    synthesized_circ = extract_circuit(mapped_circuit.num_qubits, k, cs, model)
    synthesized_circ.qasm(filename="synth_"+mapped_file_name)

    # baseline
    original_circuit = QuantumCircuit.from_qasm_file(og_circuit_filename)

    G, fs = extract_G_fs(original_circuit)
    fs, cs = zip(*fs)
    k, model = solve(original_circuit.num_qubits, G, fs)

    # print(k)
    # print(model)
    baseline_synthesized_circ = extract_circuit(original_circuit.num_qubits, k, cs, model)
    baseline_synthesized_circ.qasm(filename="baseline_synth_"+os.path.basename(og_circuit_filename))

    layout_circuit("synth_"+mapped_file_name, arch, route=True)
    layout_circuit("baseline_synth_"+os.path.basename(og_circuit_filename), arch, route=True)

    routed_baseline = QuantumCircuit.from_qasm_file("baseline_synth_"+os.path.basename(og_circuit_filename))
    routed = QuantumCircuit.from_qasm_file("synth_"+mapped_file_name)
    if routed.num_nonlocal_gates() == routed_baseline.num_nonlocal_gates():
        print(f"same {routed.num_nonlocal_gates()}")
    elif routed.num_nonlocal_gates() < routed_baseline.num_nonlocal_gates():
        print(f"baseline worse {routed.num_nonlocal_gates()} {routed_baseline.num_nonlocal_gates()}")
    else:
        print(f"baseline better {routed.num_nonlocal_gates()} {routed_baseline.num_nonlocal_gates()}")


if __name__ == "__main__":
    full_run("random_circuits/random_q4_d6.qasm", linearArch(4))