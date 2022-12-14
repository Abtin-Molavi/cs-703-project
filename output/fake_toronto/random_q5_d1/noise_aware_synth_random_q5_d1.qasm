OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.088425022) q[15];
cx q[25],q[24];
cx q[11],q[8];
