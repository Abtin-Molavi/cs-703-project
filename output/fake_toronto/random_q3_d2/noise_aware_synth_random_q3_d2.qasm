OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.0547005) q[5];
cx q[8],q[11];
cx q[8],q[5];
rz(1.139697) q[11];
