OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.6060217) q[2];
cx q[7],q[6];
cx q[9],q[8];
cx q[1],q[0];
