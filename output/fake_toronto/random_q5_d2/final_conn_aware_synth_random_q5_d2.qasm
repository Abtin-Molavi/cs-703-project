OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.6158896) q[7];
rz(0.2141631) q[2];
rz(0.48895334) q[2];
rz(1.6515954) q[0];
cx q[4],q[1];
cx q[7],q[4];
cx q[1],q[0];
