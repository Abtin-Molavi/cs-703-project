OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.088425022) q[0];
cx q[2],q[3];
cx q[25],q[24];
