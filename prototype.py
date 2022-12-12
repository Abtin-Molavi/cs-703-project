import math
import os
import signal
import subprocess
import sys
import warnings
from contextlib import contextmanager
from time import time

import numpy as np
from architectures import *
from pysat.card import CardEnc
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF, IDPool
from pysat.solvers import Solver
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.providers.fake_provider import FakeProvider
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (ApplyLayout, BasisTranslator,
                                      EnlargeWithAncilla,
                                      FullAncillaAllocation,
                                      NoiseAdaptiveLayout, SabreLayout,
                                      SabreSwap)

warnings.filterwarnings("ignore", category=DeprecationWarning) 

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

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

def fidelity_clauses(num_qubits, vpool, k, calibration_data):
    clauses = WCNF()
    for lk in range(k):
        for i in range(num_qubits):
            for j in range(num_qubits):
                for (p1,p2) in calibration_data.keys():
                        success_rate = 1-calibration_data[(p1, p2)]
                        clauses.append([-vpool.id(('q', i, lk)), -vpool.id(('t', j, lk)), -vpool.id(('m', i, p1)), -vpool.id(('m', j, p2))], weight=-1000*math.log(success_rate))
    return clauses
                        
def pretty_print_var(x, vpool, aux_vars):
        if vpool.obj(abs(x)):
            return(vpool.obj(x) if x > 0 else ("-",) + vpool.obj(-x))
        else: 
            return("aux_var")

def solve_k(num_qubits,  g_mat, fs, k, cm=None, calibration_data=None):
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
            ([] if cm is None else cnots_executable(num_qubits, vpool, k, cm) + injective_initial_map(num_qubits, vpool, cm, aux_vars) )
            )

    if calibration_data is None:

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
    else: 
        soft_clauses = fidelity_clauses(num_qubits, vpool, k, calibration_data)
        soft_clauses.extend(clauses)
        soft_clauses.to_file(f'temp_{k}.wcnf')

        lines = []
        with open(f'temp_{k}.wcnf') as f:
            lines = f.readlines()
            lines[0] = lines[0][:lines[0].find(".")]+"\n"
        with open(f'temp_{k}.wcnf', "w") as f:
            f.writelines(lines)

        subprocess.run(["../MaxHS/build/release/bin/maxhs", "-printSoln", "-no-printSoln-new-format", f"temp_{k}.wcnf"], stdout=open(f"temp_{k}.txt", "w"))

        with open(f"temp_{k}.txt") as f:
            lines = f.readlines()
            for line in lines:
                if "UNSATISFIABLE" in line:
                    return False, None
                if line.startswith("v "):
                    model = line.split(" ")[1:]
                    return True, [vpool.obj(int(x)) for x in model if int(x) > 0]

def solve(num_qubits,  g_mat, fs, timeout, cm=None, calibration_data=None):
    try:
        with time_limit(timeout):
            solved = False
            k = 0
            while not solved:
                solved, model = solve_k(num_qubits,  g_mat, fs, k, cm, calibration_data)
                k += 1
            return k-1, model
    except TimeoutException as e:
        return None, None

def extract_circuit(num_qubits, k, cs, model):
    mapping = {}
    for v in model:
        if v is None: continue
        if v[0] == "m":
            mapping[v[1]] = v[2]
    
    circuit = QuantumCircuit(num_qubits)
    applied_rz = [False] * len(cs)
    for lk in range(k+1):
        c = None
        t = None
        for v in model:
            if v is None: continue
            if v[2] == lk:
                if v[0] == "q":
                    c = mapping[v[1]] if v[1] in mapping.keys() else v[1]
                if v[0] == "t":
                    t = mapping[v[1]] if v[1] in mapping.keys() else v[1]
                if v[0] == "p" and not applied_rz[v[3]]:
                    circuit.rz(cs[v[3]], mapping[v[1]] if v[1] in mapping.keys() else v[1])
                    applied_rz[v[3]] = True
        if lk < k:
            circuit.cx(c, t)
    return circuit

def layout_circuit(file_name, backend, noise_aware=False):
    circ = QuantumCircuit.from_qasm_file(file_name)
    cm = CouplingMap(backend.configuration().coupling_map)
    
    if not noise_aware:
        pm = PassManager([SabreLayout(coupling_map=cm, routing_pass=SabreSwap(cm, heuristic="lookahead")), FullAncillaAllocation(cm),
                      EnlargeWithAncilla(), ApplyLayout(), SabreSwap(coupling_map=cm, heuristic="lookahead"), BasisTranslator(sel, ['rz', 'cx'])]) 
        return pm.run(circ)
    else:
        pm = PassManager([NoiseAdaptiveLayout(backend.properties()), FullAncillaAllocation(cm),
                      EnlargeWithAncilla(), ApplyLayout(), SabreSwap(coupling_map=cm, heuristic="lookahead"), BasisTranslator(sel, ['rz', 'cx'])]) 
        return pm.run(circ)

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

def to_adjacency_matrix(backend):
    mat = np.zeros((backend.configuration().n_qubits, backend.configuration().n_qubits))
    for (u,v) in backend.configuration().coupling_map:
        mat[u][v] = 1
    return mat

def get_calibration_data(backend):
    calibration_data = {}
    for edge in backend.configuration().coupling_map:
        calibration_data[tuple(edge)] = backend.properties().gate_error('cx', edge)
    return calibration_data

def get_fidelity(circuit, calibration_data):
    fidelity = 1
    for g in circuit.data:
        if g.operation.name == "cx":
            c = g.qubits[0].index
            t = g.qubits[1].index
            fidelity *= 1-calibration_data[(c, t)]
    return fidelity

def check_connectivity(file, cm):
  with open(file) as f:
    lines = f.readlines()
    for line in lines:
      if "cx" in line:
        control = int(line[line.find("[")+1:line.find("]")])
        target = int(line[line.rfind("[")+1:line.rfind("]")])
        if (control, target) not in cm:
          return False
  return True

def full_run(og_circuit_filename, backend_output_dir, backend, timeout):
    input_circuit = QuantumCircuit.from_qasm_file(og_circuit_filename)
    base_name = os.path.basename(og_circuit_filename)
    arch = to_adjacency_matrix(backend)
    calibration_data = get_calibration_data(backend)
    result = {}
    result["circuit"] = os.path.basename(og_circuit_filename)
    result["backend"] = backend.name()

    benchmark_output_dir = f"{backend_output_dir}/{os.path.splitext(base_name)[0]}"
    if not os.path.exists(benchmark_output_dir):
        os.makedirs(benchmark_output_dir)

    # noise-aware
    time1 = time()
    G, fs = extract_G_fs(input_circuit)
    fs, cs = zip(*fs)
    k, model = solve(input_circuit.num_qubits, G, fs, timeout, arch, calibration_data)

    if k is not None:
        synthesized_circ = extract_circuit(backend.configuration().n_qubits, k, cs, model)
        time2 = time()
        time_na = time2 - time1
        noise_aware_synth_filename = f"{benchmark_output_dir}/noise_aware_synth_"+base_name
        synthesized_circ.qasm(filename=noise_aware_synth_filename)
        final_circuit_na = layout_circuit(noise_aware_synth_filename, backend, noise_aware=True) if not check_connectivity(noise_aware_synth_filename, backend.configuration().coupling_map) else synthesized_circ
        final_circuit_na.qasm(filename=f"{benchmark_output_dir}/final_noise_aware_synth_{base_name}")
        result["noise_aware"] = {"cx": final_circuit_na.num_nonlocal_gates(), "fidelity": get_fidelity(final_circuit_na, calibration_data), "time": time_na}
    else:
        result["noise_aware"] = {"timeout": True}
    
    # connectivity aware
    time3 = time()
    G, fs = extract_G_fs(input_circuit)
    fs, cs = zip(*fs)
    k, model = solve(input_circuit.num_qubits, G, fs, timeout, arch)

    if k is not None:
        synthesized_circ = extract_circuit(backend.configuration().n_qubits, k, cs, model)
        time4 = time()
        time_ca = time4 - time3
        conn_aware_synth_filename = f"{benchmark_output_dir}/conn_aware_synth_"+base_name
        synthesized_circ.qasm(filename=conn_aware_synth_filename)
        final_circuit_ca = layout_circuit(conn_aware_synth_filename, backend) if not check_connectivity(conn_aware_synth_filename, backend.configuration().coupling_map) else synthesized_circ
        final_circuit_ca.qasm(filename=f"{benchmark_output_dir}/final_conn_aware_synth_{base_name}")
        result["conn_aware"] = {"cx": final_circuit_ca.num_nonlocal_gates(), "fidelity": get_fidelity(final_circuit_ca, calibration_data), "time": time_ca}
    else:
        result["conn_aware"] = {"timeout": True}

    # baseline
    time5 = time()
    G, fs = extract_G_fs(input_circuit)
    fs, cs = zip(*fs)
    k, model = solve(input_circuit.num_qubits, G, fs, timeout)

    if k is not None:
        baseline_synthesized_circ = extract_circuit(backend.configuration().n_qubits, k, cs, model)
        time6 = time()
        time_baseline = time6 - time5
        baseline_synth_filename = f"{benchmark_output_dir}/baseline_synth_"+base_name
        baseline_synthesized_circ.qasm(filename=baseline_synth_filename)
        final_circuit_baseline = layout_circuit(baseline_synth_filename, backend) if not check_connectivity(baseline_synth_filename, backend.configuration().coupling_map) else synthesized_circ
        final_circuit_baseline.qasm(filename=f"{benchmark_output_dir}/final_baseline_synth_{base_name}")
        final_cicuit_baseline_na = layout_circuit(baseline_synth_filename, backend, noise_aware=True)
        final_cicuit_baseline_na.qasm(filename=f"{benchmark_output_dir}/final_baseline_na_synth_{base_name}")
        result["baseline"] = {"cx": final_circuit_baseline.num_nonlocal_gates(), "fidelity": get_fidelity(final_circuit_baseline, calibration_data), "time": time_baseline}
        result["baseline_na"] = {"cx": final_cicuit_baseline_na.num_nonlocal_gates(), "fidelity": get_fidelity(final_cicuit_baseline_na, calibration_data), "time": time_baseline}
    else:
        result["baseline"] = {"timeout": True}
        result["baseline_na"] = {"timeout": True}

    with open(f"{benchmark_output_dir}/results.txt", "w") as f:
        f.write(f"{result}")


if __name__ == "__main__":
    args = sys.argv[1:]
    benchmark_dir = args[0]
    output_dir = args[1]
    architecture = args[2]
    timeout = int(args[3])
    provider = FakeProvider()
    backend = provider.get_backend(architecture)
    backend_output_dir = f"{output_dir}/{backend.name()}"
    if not os.path.exists(backend_output_dir):
        os.makedirs(backend_output_dir)

    for filename in os.listdir(benchmark_dir):
        full_run(f"{benchmark_dir}/{filename}", backend_output_dir, backend, timeout)