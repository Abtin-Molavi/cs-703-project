OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.6060217) q[21];
cx q[11],q[8];
cx q[25],q[24];
cx q[7],q[6];
