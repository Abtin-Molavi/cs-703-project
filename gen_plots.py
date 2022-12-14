import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FormatStrFormatter
from qiskit.circuit import QuantumCircuit

sns.set_theme(palette="colorblind", font_scale=2.5)
sns.set_style("ticks",{'font.family': 'serif', 'axes.grid' : True})

CONN_AWARE = "conn_aware"
NOISE_AWARE = "noise_aware"
BASELINE = "baseline"
BASELINE_NA = "baseline_na"

US_CA = "WOQ-CA"
US_NA = "WOQ-HA"
US_BASE = "WOQ"

MANILA = "IBM Manila"
TORONTO = "IBM Toronto"

FIDELITY_COLOR = "#e7e7eb"

# benchmarks = [
#     'random_q1_d1', 'random_q1_d2', 'random_q1_d3', 'random_q1_d4', 'random_q1_d5', 'random_q1_d6', 'random_q1_d7', 'random_q1_d8', 'random_q1_d9', 'random_q1_d10', 'random_q1_d11', 'random_q1_d12', 'random_q1_d13', 'random_q1_d14', 'random_q1_d15', 'random_q1_d16', 
#     'random_q2_d1', 'random_q2_d2', 'random_q2_d3', 'random_q2_d4', 'random_q2_d5', 'random_q2_d6', 'random_q2_d7', 'random_q2_d8', 'random_q2_d9', 'random_q2_d10', 'random_q2_d11', 'random_q2_d12', 'random_q2_d13', 'random_q2_d14', 'random_q2_d15', 'random_q2_d16',
#     'random_q3_d1', 'random_q3_d2', 'random_q3_d3', 'random_q3_d4', 'random_q3_d5', 'random_q3_d6', 'random_q3_d7', 'random_q3_d8', 'random_q3_d9', 'random_q3_d10', 'random_q3_d11', 'random_q3_d12', 'random_q3_d13', 'random_q3_d14', 'random_q3_d15', 'random_q3_d16',
#     'random_q4_d1', 'random_q4_d2', 'random_q4_d3', 'random_q4_d4', 'random_q4_d5', 'random_q4_d6', 'random_q4_d7', 'random_q4_d8', 'random_q4_d9', 'random_q4_d10', 'random_q4_d11', 'random_q4_d12', 'random_q4_d13', 'random_q4_d14', 'random_q4_d15', 'random_q4_d16',
#     'random_q5_d1', 'random_q5_d2', 'random_q5_d3', 'random_q5_d4', 'random_q5_d5', 'random_q5_d6', 'random_q5_d7', 'random_q5_d8', 'random_q5_d9', 'random_q5_d10', 'random_q5_d11', 'random_q5_d12', 'random_q5_d13', 'random_q5_d16'
# ]

# benchmarks_toronto = [
#     'random_q1_d1', 'random_q1_d2', 'random_q1_d3', 'random_q1_d4', 'random_q1_d5', 'random_q1_d6', 'random_q1_d7', 'random_q1_d8', 'random_q1_d9', 'random_q1_d10', 'random_q1_d11', 'random_q1_d12', 'random_q1_d13', 'random_q1_d14', 'random_q1_d15', 'random_q1_d16',
#     'random_q2_d1', 'random_q2_d2', 'random_q2_d3', 'random_q2_d4', 'random_q2_d5', 'random_q2_d6', 'random_q2_d7', 'random_q2_d8', 'random_q2_d9', 'random_q2_d10', 'random_q2_d11', 'random_q2_d12', 'random_q2_d13', 'random_q2_d14', 'random_q2_d15', 'random_q2_d16',
#     'random_q3_d1', 'random_q3_d2', 'random_q3_d3', 'random_q3_d4', 'random_q3_d5', 'random_q3_d6', 'random_q3_d7', 'random_q3_d8', 'random_q3_d9', 'random_q3_d10', 'random_q3_d11', 'random_q3_d12', 'random_q3_d13', 'random_q3_d14', 'random_q3_d15', 'random_q3_d16',
#     'random_q4_d1', 'random_q4_d2', 'random_q4_d3', 'random_q4_d4', 'random_q4_d5', 'random_q4_d6', 'random_q4_d7', 'random_q4_d8', 'random_q4_d9', 'random_q4_d10', 'random_q4_d11', 'random_q4_d12', 'random_q4_d13', 'random_q4_d14', 'random_q4_d15', 'random_q4_d16',
#     'random_q5_d1', 'random_q5_d2', 'random_q5_d3', 'random_q5_d4', 'random_q5_d5', 'random_q5_d6', 'random_q5_d7', 'random_q5_d8', 'random_q5_d9', 'random_q5_d10', 'random_q5_d11', 'random_q5_d12', 'random_q5_d13',
#     'random_q6_d1', 'random_q6_d2', 'random_q6_d3', 'random_q6_d4', 'random_q6_d5', 'random_q6_d6', 'random_q6_d7', 'random_q6_d8', 'random_q6_d9', 'random_q6_d10',
#     'random_q7_d1', 'random_q7_d2', 'random_q7_d3', 'random_q7_d4', 'random_q7_d5', 'random_q7_d6', 'random_q7_d7', 'random_q7_d8', 'random_q7_d9', 
#     'random_q8_d1', 'random_q8_d2', 'random_q8_d3', 'random_q8_d4', 'random_q8_d5', 'random_q8_d6', 'random_q8_d7', 'random_q8_d8', 
#     'random_q9_d1', 'random_q9_d2', 'random_q9_d3', 'random_q9_d4', 'random_q9_d5', 'random_q9_d6', 'random_q9_d7', 'random_q9_d8',
#     'random_q10_d1', 'random_q10_d2', 'random_q10_d3', 'random_q10_d4', 'random_q10_d5', 'random_q10_d6', 'random_q10_d7', 
#     'random_q11_d1', 'random_q11_d2', 'random_q11_d3', 'random_q11_d4', 'random_q11_d5', 'random_q11_d6', 
#     'random_q12_d1', 'random_q12_d2', 'random_q12_d3', 'random_q12_d4', 'random_q12_d5', 'random_q13_d1', 
#     'random_q13_d2', 'random_q13_d3', 'random_q13_d4', 'random_q13_d5', 
#     'random_q14_d1', 'random_q14_d2', 'random_q14_d3', 'random_q14_d4', 
#     'random_q15_d1', 'random_q15_d2', 'random_q15_d3', 'random_q15_d4', 
#     'random_q16_d1', 'random_q16_d2', 'random_q16_d3'
# ]

def parse_results(dir):
    results = {}
    for file in os.listdir(dir):
        with open(f"{dir}/{file}/results.txt") as f:
            results[file] = ast.literal_eval(f.readline())
    return results

def value(results_dict, benchmark, config_name):
    return results_dict[benchmark][config_name]["cx"] if "timeout" not in results_dict[benchmark][config_name] else -1

def get_ratios(x, y, results_dict, metric):
    ratios = {}
    for b in results_dict.keys():
        if "timeout" not in results_dict[b][x] and "timeout" not in results_dict[b][y]:
            ratios[b] = results_dict[b][y][metric]-results_dict[b][x][metric]# if results_dict[b][x]["cx"] != 0 else 1 if results_dict[b][y]["cx"] == 0 else -100
    return ratios

def get_timeouts(x, y, results_dict, val):
    timeouts = {}
    for b in results_dict.keys():
        if "timeout" in results_dict[b][x] and "timeout" in results_dict[b][y]:
            timeouts[b] = 0
        elif "timeout" in results_dict[b][x]:
            timeouts[b] = val
        elif "timeout" in results_dict[b][y]:
            raise RuntimeError()

    return timeouts

def get_cactus_data(config_name, results_dict):
    x = []
    y = []
    temp = {k:v for k,v in results_dict.items() if "timeout" not in v[config_name]}
    for b in sorted(list(temp.keys()), key=lambda x: temp[x][config_name]["time"]):
        if "timeout" not in temp[b][config_name]:
            x.append(len(x)+1)
            y.append(y[-1]+temp[b][config_name]["time"]) if len(y) > 0 else y.append(temp[b][config_name]["time"])

    return x,y

def get_largest_circ_solved(config_name, results_dict):
    result = 0
    for b in results_dict.keys():
        if "timeout" not in results_dict[b][config_name]:
            result = max(result, QuantumCircuit.from_qasm_file(f"random_circuits/{b}.qasm").num_nonlocal_gates())
    return result

if __name__ == "__main__":
    fake_manila_data = parse_results("output/fake_manila")

    ratios = get_ratios(CONN_AWARE, BASELINE, fake_manila_data, "cx")

    data = pd.DataFrame({
        "program" : sorted(list(ratios.keys()), key=ratios.get),
        "ratio" : sorted(list(ratios.values())),
        # US_CA : [fake_manila_data[b][CONN_AWARE]["cx"] for b in benchmarks if "timeout" not in fake_manila_data[b][CONN_AWARE] and "timeout" not in fake_manila_data[b][BASELINE]], 
        # US_BASE : [fake_manila_data[b][BASELINE]["cx"] for b in benchmarks if "timeout" not in fake_manila_data[b][CONN_AWARE] and "timeout" not in fake_manila_data[b][BASELINE]],
        "type" : ["no timeout"] * len(ratios)
    })

    timeouts = get_timeouts(CONN_AWARE, BASELINE, fake_manila_data, -10.45)

    timeoutData = pd.DataFrame({
        "program" : sorted(list(timeouts.keys()), key=timeouts.get),
        "ratio" : sorted(list(timeouts.values())),
        # US_CA : [value(fake_manila_data, b, CONN_AWARE) for b in benchmarks if "timeout" in fake_manila_data[b][CONN_AWARE] or "timeout" in fake_manila_data[b][BASELINE]], 
        # US_BASE : [value(fake_manila_data, b, BASELINE) for b in benchmarks if "timeout" in fake_manila_data[b][CONN_AWARE] or "timeout" in fake_manila_data[b][BASELINE]],
        "type": ["timeout"] * len(timeouts)
    })

    scatter_timeout = sns.scatterplot(
        data = timeoutData,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#de8f05"],
        markers = ["s"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    scatter = sns.scatterplot(
        data=data,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    x = np.arange(0, len(ratios.keys()) + len(timeouts.keys()), 1)

    scatter.figure.set_size_inches(10, 10)
    scatter.grid(False, axis="x")
    scatter.set(xlabel = None, xticklabels=[])
    scatter.tick_params(bottom=False)
    plt.gca().set_box_aspect(1)
    l = plt.gca().legend(title="status", title_fontsize='small', fontsize='small', loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    plt.plot(x,[0]*len(x), "black", linestyle="dashed", linewidth=4)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    print(yabs_max)
    plt.gca().set_ylim(ymin=-10.5, ymax=10.5)
    plt.tight_layout()
    scatter.set(xlabel="", ylabel=r"$\bf{" + "CNOT" + "}$" + " Difference", title=f"{US_CA} vs. {US_BASE} ({MANILA})")
    scatter.figure.savefig(f"rq1_manila_ca.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close()

    ############################################################################

    ratios = get_ratios(NOISE_AWARE, BASELINE_NA, fake_manila_data, "fidelity")
    ratios = {k:-v for k,v in ratios.items()}

    data = pd.DataFrame({
        "program" : sorted(list(ratios.keys()), key=ratios.get),
        "ratio" : sorted(list(ratios.values())),
        "type" : ["no timeout"] * len(ratios)
    })

    timeouts = get_timeouts(NOISE_AWARE, BASELINE_NA, fake_manila_data, -0.111)

    timeoutData = pd.DataFrame({
        "program" : sorted(list(timeouts.keys()), key=timeouts.get),
        "ratio" : sorted(list(timeouts.values())),
        "type": ["timeout"] * len(timeouts)
    })

    scatter_timeout = sns.scatterplot(
        data = timeoutData,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#de8f05"],
        markers = ["s"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    scatter = sns.scatterplot(
        data=data,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    x = np.arange(0, len(ratios.keys()) + len(timeouts.keys()), 1)

    scatter.figure.set_size_inches(10, 10)
    scatter.grid(False, axis="x")
    scatter.set(xlabel = None, xticklabels=[])
    scatter.tick_params(bottom=False)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().set_box_aspect(1)
    plt.gca().set_facecolor(FIDELITY_COLOR)
    l = plt.gca().legend(title="status", title_fontsize='small', fontsize='small', loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    plt.plot(x,[0]*len(x), "black", linestyle="dashed", linewidth=4)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    print(yabs_max)
    plt.gca().set_ylim(ymin=-0.1114, ymax=0.1114)
    plt.tight_layout()
    scatter.set(xlabel="", ylabel=r"$\bf{" + "Fidelity" + "}$" + " Difference", title=f"{US_NA} vs. {US_BASE} ({MANILA})")
    scatter.figure.savefig(f"rq1_manila_ha.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close() 

    ############################################################################

    ratios = get_ratios(NOISE_AWARE, CONN_AWARE, fake_manila_data, "fidelity")
    ratios = {k:-v for k,v in ratios.items()}

    data = pd.DataFrame({
        "program" : sorted(list(ratios.keys()), key=ratios.get),
        "ratio" : sorted(list(ratios.values())),
        "type" : ["no timeout"] * len(ratios)
    })

    timeouts = get_timeouts(NOISE_AWARE, CONN_AWARE, fake_manila_data, -0.03785)

    timeoutData = pd.DataFrame({
        "program" : sorted(list(timeouts.keys()), key=timeouts.get),
        "ratio" : sorted(list(timeouts.values())),
        "type": ["timeout"] * len(timeouts)
    })

    scatter_timeout = sns.scatterplot(
        data = timeoutData,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#de8f05"],
        markers = ["s"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    scatter = sns.scatterplot(
        data=data,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    x = np.arange(0, len(ratios.keys()) + len(timeouts.keys()), 1)

    scatter.figure.set_size_inches(10, 10)
    scatter.grid(False, axis="x")
    scatter.set(xlabel = None, xticklabels=[])
    scatter.tick_params(bottom=False)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().set_box_aspect(1)
    plt.gca().set_facecolor(FIDELITY_COLOR)
    l = plt.gca().legend(title="status", title_fontsize='small', fontsize='small', loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    plt.plot(x,[0]*len(x), "black", linestyle="dashed", linewidth=4)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    print(yabs_max)
    plt.gca().set_ylim(ymin=-0.0379, ymax=0.0379)
    plt.tight_layout()
    scatter.set(xlabel="", ylabel=r"$\bf{" + "Fidelity" + "}$" + " Difference", title=f"{US_NA} vs. {US_CA} ({MANILA})")
    scatter.figure.savefig(f"rq3_manila.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close()

    ############################################################################

    baseline_x, baseline_y = get_cactus_data(BASELINE, fake_manila_data)
    ca_x, ca_y = get_cactus_data(CONN_AWARE, fake_manila_data)
    na_x, na_y = get_cactus_data(NOISE_AWARE, fake_manila_data)

    baseline_data = pd.DataFrame({
        "num_solved" : baseline_x,
        "time" : baseline_y,
        "config" : [US_BASE] * len(baseline_x),
    })

    ca_data = pd.DataFrame({
        "num_solved" : ca_x,
        "time" : ca_y,
        "config" : [US_CA] * len(ca_x)
    })

    na_data = pd.DataFrame({
        "num_solved" : na_x,
        "time" : na_y,
        "config" : [US_NA] * len(na_x)
    })

    scatter1 = sns.scatterplot(
        data = baseline_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "config",
        s = 300,
        linewidth=0.01
    )

    sns.lineplot(
        data = baseline_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#0173b2"],
    )

    scatter2 = sns.scatterplot(
        data = ca_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#de8f05"],
        markers = ["^"],
        style = "config",
        s = 300,
        linewidth=0.01
    )

    sns.lineplot(
        data = ca_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#de8f05"],
    )

    scatter3 = sns.scatterplot(
        data = na_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#029e73"],
        markers = ["x"],
        style = "config",
        s = 300,
        linewidth=0.01
    )

    sns.lineplot(
        data = na_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#029e73"],
    )

    scatter3.figure.set_size_inches(10, 10)
    handles, labels = plt.gca().get_legend_handles_labels()
    l = plt.gca().legend([(handles[0], handles[1]), (handles[2], handles[3]), (handles[4], handles[5])], [labels[0],labels[2],labels[4]], handlelength=3,
          handler_map={tuple: HandlerTuple(ndivide=None)}, title="", fontsize="small", loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    l.legendHandles[2]._sizes = [300]
    scatter3.grid(False, axis="x")
    scatter3.grid(False, axis="y")
    scatter3.set(xlabel="# Benchmarks Solved", ylabel="Time (s)", title=f"{MANILA}")
    scatter3.figure.savefig(f"rq2_manila_cactus.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close()

    ############################################################################

    data = pd.DataFrame({
        "cnots" : [get_largest_circ_solved(BASELINE, fake_manila_data), get_largest_circ_solved(CONN_AWARE, fake_manila_data), get_largest_circ_solved(NOISE_AWARE, fake_manila_data)],
        "method" : [US_BASE, US_CA, US_NA]
    })
    print(get_largest_circ_solved(BASELINE, fake_manila_data))

    barchart = sns.catplot(
        data=data,
        x = "method",
        y = "cnots",
        hue = "method",
        kind = "bar",
        dodge = False
    )
    barchart.set(ylabel = "Largest Circuit Solved (# CNOT gates)", xlabel="", title=MANILA)

    barchart.savefig("rq2_manila_bar.pdf")

    plt.close()

    ################################### TORONTO ################################
    fake_toronto_data = parse_results("output/fake_toronto")

    ratios = get_ratios(CONN_AWARE, BASELINE, fake_toronto_data, "cx")

    data = pd.DataFrame({
        "program" : sorted(list(ratios.keys()), key=ratios.get),
        "ratio" : sorted(list(ratios.values())),
        "type" : ["no timeout"] * len(ratios)
    })

    timeouts = get_timeouts(CONN_AWARE, BASELINE, fake_toronto_data, -14.65)

    timeoutData = pd.DataFrame({
        "program" : sorted(list(timeouts.keys()), key=timeouts.get),
        "ratio" : sorted(list(timeouts.values())),
        "type": ["timeout"] * len(timeouts)
    })

    scatter_timeout = sns.scatterplot(
        data = timeoutData,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#de8f05"],
        markers = ["s"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    scatter = sns.scatterplot(
        data=data,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    x = np.arange(0, len(ratios.keys()) + len(timeouts.keys()), 1)

    scatter.figure.set_size_inches(10, 10)
    scatter.grid(False, axis="x")
    scatter.set(xlabel = None, xticklabels=[])
    scatter.tick_params(bottom=False)
    plt.gca().set_box_aspect(1)
    l = plt.gca().legend(title="status", title_fontsize='small', fontsize='small', loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    plt.plot(x,[0]*len(x), "black", linestyle="dashed", linewidth=4)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    print(yabs_max)
    plt.gca().set_ylim(ymin=-14.7, ymax=14.7)
    plt.tight_layout()
    scatter.set(xlabel="", ylabel=r"$\bf{" + "CNOT" + "}$" + " Difference", title=f"{US_CA} vs. {US_BASE} ({TORONTO})")
    scatter.figure.savefig(f"rq1_toronto_ca.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close()

    ############################################################################

    ratios = get_ratios(NOISE_AWARE, BASELINE_NA, fake_toronto_data, "fidelity")
    ratios = {k:-v for k,v in ratios.items()}

    data = pd.DataFrame({
        "program" : sorted(list(ratios.keys()), key=ratios.get),
        "ratio" : sorted(list(ratios.values())),
        "type" : ["no timeout"] * len(ratios)
    })

    timeouts = get_timeouts(NOISE_AWARE, BASELINE_NA, fake_toronto_data, -0.19)

    timeoutData = pd.DataFrame({
        "program" : sorted(list(timeouts.keys()), key=timeouts.get),
        "ratio" : sorted(list(timeouts.values())),
        "type": ["timeout"] * len(timeouts)
    })

    scatter_timeout = sns.scatterplot(
        data = timeoutData,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#de8f05"],
        markers = ["s"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    scatter = sns.scatterplot(
        data=data,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    x = np.arange(0, len(ratios.keys()) + len(timeouts.keys()), 1)

    scatter.figure.set_size_inches(10, 10)
    scatter.grid(False, axis="x")
    scatter.set(xlabel = None, xticklabels=[])
    scatter.tick_params(bottom=False)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().set_box_aspect(1)
    plt.gca().set_facecolor(FIDELITY_COLOR)
    l = plt.gca().legend(title="status", title_fontsize='small', fontsize='small', loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    plt.plot(x,[0]*len(x), "black", linestyle="dashed", linewidth=4)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    print(yabs_max)
    plt.gca().set_ylim(ymin=-0.1907818003968657, ymax=0.1907818003968657)
    plt.tight_layout()
    scatter.set(xlabel="", ylabel=r"$\bf{" + "Fidelity" + "}$" + " Difference", title=f"{US_NA} vs. {US_BASE} ({TORONTO})")
    scatter.figure.savefig(f"rq1_toronto_ha.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close() 

    ############################################################################

    ratios = get_ratios(NOISE_AWARE, CONN_AWARE, fake_toronto_data, "fidelity")
    ratios = {k:-v for k,v in ratios.items()}

    data = pd.DataFrame({
        "program" : sorted(list(ratios.keys()), key=ratios.get),
        "ratio" : sorted(list(ratios.values())),
        "type" : ["no timeout"] * len(ratios)
    })

    timeouts = get_timeouts(NOISE_AWARE, CONN_AWARE, fake_toronto_data, -0.384)

    timeoutData = pd.DataFrame({
        "program" : sorted(list(timeouts.keys()), key=timeouts.get),
        "ratio" : sorted(list(timeouts.values())),
        "type": ["timeout"] * len(timeouts)
    })

    scatter_timeout = sns.scatterplot(
        data = timeoutData,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#de8f05"],
        markers = ["s"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    scatter = sns.scatterplot(
        data=data,
        y = "ratio",
        x = "program",
        hue = "type",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "type",
        s = 300,
        linewidth=0.01
    )

    x = np.arange(0, len(ratios.keys()) + len(timeouts.keys()), 1)

    scatter.figure.set_size_inches(10, 10)
    scatter.grid(False, axis="x")
    scatter.set(xlabel = None, xticklabels=[])
    scatter.tick_params(bottom=False)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().set_box_aspect(1)
    plt.gca().set_facecolor(FIDELITY_COLOR)
    l = plt.gca().legend(title="status", title_fontsize='small', fontsize='small', loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    plt.plot(x,[0]*len(x), "black", linestyle="dashed", linewidth=4)
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    print(yabs_max)
    plt.gca().set_ylim(ymin=-0.3841909040452157, ymax=0.3841909040452157)
    plt.tight_layout()
    scatter.set(xlabel="", ylabel=r"$\bf{" + "Fidelity" + "}$" + " Difference", title=f"{US_NA} vs. {US_CA} ({TORONTO})")
    scatter.figure.savefig(f"rq3_toronto.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close()

    ############################################################################

    baseline_x, baseline_y = get_cactus_data(BASELINE, fake_toronto_data)
    ca_x, ca_y = get_cactus_data(CONN_AWARE, fake_toronto_data)
    na_x, na_y = get_cactus_data(NOISE_AWARE, fake_toronto_data)

    baseline_data = pd.DataFrame({
        "num_solved" : baseline_x,
        "time" : baseline_y,
        "config" : [US_BASE] * len(baseline_x),
    })

    ca_data = pd.DataFrame({
        "num_solved" : ca_x,
        "time" : ca_y,
        "config" : [US_CA] * len(ca_x)
    })

    na_data = pd.DataFrame({
        "num_solved" : na_x,
        "time" : na_y,
        "config" : [US_NA] * len(na_x)
    })

    scatter1 = sns.scatterplot(
        data = baseline_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#0173b2"],
        markers = ["o"],
        style = "config",
        s = 300,
        linewidth=0.01
    )

    sns.lineplot(
        data = baseline_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#0173b2"],
    )

    scatter2 = sns.scatterplot(
        data = ca_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#de8f05"],
        markers = ["^"],
        style = "config",
        s = 300,
        linewidth=0.01
    )

    sns.lineplot(
        data = ca_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#de8f05"],
    )

    scatter3 = sns.scatterplot(
        data = na_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#029e73"],
        markers = ["x"],
        style = "config",
        s = 300,
        linewidth=0.01
    )

    sns.lineplot(
        data = na_data,
        y = "time",
        x = "num_solved",
        hue = "config",
        palette = ["#029e73"],
    )

    scatter3.figure.set_size_inches(10, 10)
    handles, labels = plt.gca().get_legend_handles_labels()
    l = plt.gca().legend([(handles[0], handles[1]), (handles[2], handles[3]), (handles[4], handles[5])], [labels[0],labels[2],labels[4]], handlelength=3,
          handler_map={tuple: HandlerTuple(ndivide=None)}, title="", fontsize="small", loc="upper left")
    l.legendHandles[0]._sizes = [300]
    l.legendHandles[1]._sizes = [300]
    l.legendHandles[2]._sizes = [300]
    scatter3.grid(False, axis="x")
    scatter3.grid(False, axis="y")
    scatter3.set(xlabel="# Benchmarks Solved", ylabel="Time (s)", title=f"{TORONTO}")
    scatter3.figure.savefig(f"rq2_toronto_cactus.pdf", bbox_inches='tight', pad_inches=0.01) 
    plt.close()

    ############################################################################

    data = pd.DataFrame({
        "cnots" : [get_largest_circ_solved(BASELINE, fake_toronto_data), get_largest_circ_solved(CONN_AWARE, fake_toronto_data), get_largest_circ_solved(NOISE_AWARE, fake_toronto_data)],
        "method" : [US_BASE, US_CA, US_NA]
    })
    print(get_largest_circ_solved(BASELINE, fake_toronto_data))

    barchart = sns.catplot(
        data=data,
        x = "method",
        y = "cnots",
        hue = "method",
        kind = "bar",
        dodge = False
    )
    barchart.set(ylabel = "Largest Circuit Solved (# CNOT gates)", xlabel="", title=TORONTO)

    barchart.savefig("rq2_toronto_bar.pdf")
    
    plt.close()

