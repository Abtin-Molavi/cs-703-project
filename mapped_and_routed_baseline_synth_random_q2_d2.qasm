OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.7189803) q[14];
rz(5.6439883) q[16];
cx q[14],q[16];
