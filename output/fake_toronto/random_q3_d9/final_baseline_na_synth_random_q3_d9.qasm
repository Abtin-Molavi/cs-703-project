OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.2860452) q[24];
rz(2.4583473) q[24];
rz(3.4641986) q[24];
rz(5.4178612) q[24];
rz(4.7529247) q[24];
rz(1.035201) q[26];
rz(5.2878801) q[26];
rz(5.3628989) q[26];
cx q[26],q[25];
rz(0.1664508) q[25];
rz(2.454825) q[25];
rz(6.0495409) q[25];
rz(5.2625473) q[25];
cx q[24],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
rz(3.3119823) q[24];
cx q[26],q[25];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
rz(0.19116485) q[25];
rz(5.7365978) q[25];