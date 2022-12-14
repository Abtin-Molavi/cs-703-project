OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6439883) q[8];
rz(4.7189803) q[11];
cx q[11],q[8];
