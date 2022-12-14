OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[25],q[24];
rz(1.139697) q[24];
rz(4.0547005) q[26];
cx q[25],q[26];
