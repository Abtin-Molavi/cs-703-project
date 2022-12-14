OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6439883) q[0];
rz(4.7189803) q[1];
cx q[1],q[0];
