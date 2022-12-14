OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[1],q[2];
rz(pi/4) q[2];
cx q[0],q[1];
