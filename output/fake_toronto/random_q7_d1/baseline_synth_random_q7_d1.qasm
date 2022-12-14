OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.6060217) q[4];
cx q[3],q[6];
cx q[0],q[1];
cx q[2],q[5];
