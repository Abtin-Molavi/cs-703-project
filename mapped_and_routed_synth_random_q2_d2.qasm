OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6439883) q[12];
rz(4.7189803) q[13];
cx q[13],q[12];
