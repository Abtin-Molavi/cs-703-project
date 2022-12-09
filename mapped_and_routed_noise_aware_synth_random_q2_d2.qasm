OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6439883) q[24];
cx q[25],q[24];
rz(4.7189803) q[25];
