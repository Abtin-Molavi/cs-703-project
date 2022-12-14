OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.99103) q[0];
cx q[24],q[25];
