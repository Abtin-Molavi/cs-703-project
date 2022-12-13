OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.0547005) q[2];
cx q[1],q[0];
rz(1.139697) q[0];
cx q[1],q[2];
