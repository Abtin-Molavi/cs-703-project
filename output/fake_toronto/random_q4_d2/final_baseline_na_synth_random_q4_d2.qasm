OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.6904972) q[0];
rz(5.4838375) q[0];
rz(3.773203) q[24];
cx q[25],q[26];
cx q[24],q[25];
rz(0.65867831) q[26];
