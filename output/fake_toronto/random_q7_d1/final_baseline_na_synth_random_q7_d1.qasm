OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.6060217) q[0];
cx q[2],q[3];
cx q[11],q[8];
cx q[24],q[25];
