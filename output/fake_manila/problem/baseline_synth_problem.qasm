OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[0],q[1];
rz(pi/4) q[1];
cx q[2],q[0];
