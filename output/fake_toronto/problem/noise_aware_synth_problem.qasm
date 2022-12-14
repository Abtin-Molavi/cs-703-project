OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[8],q[11];
cx q[5],q[8];
rz(pi/4) q[11];
