OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.0547005) q[2];
cx q[3],q[4];
cx q[3],q[2];
rz(1.139697) q[4];
