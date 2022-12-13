OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(5.6439883) q[3];
rz(4.7189803) q[4];
cx q[4],q[3];
