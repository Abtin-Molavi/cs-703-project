OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.2141631) q[0];
rz(0.48895334) q[0];
rz(3.6158896) q[22];
cx q[25],q[24];
cx q[22],q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(1.6515954) q[26];
cx q[25],q[26];