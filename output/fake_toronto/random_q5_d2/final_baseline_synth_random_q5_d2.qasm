OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.2141631) q[4];
rz(0.48895334) q[4];
rz(1.6515954) q[11];
cx q[12],q[13];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
rz(3.6158896) q[15];
cx q[15],q[12];
