OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[5],q[4];
cx q[1],q[3];
cx q[2],q[0];
