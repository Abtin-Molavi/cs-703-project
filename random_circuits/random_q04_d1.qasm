OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(0.64783398) q[0];
rz(5.6790115) q[1];
cx q[2],q[3];
