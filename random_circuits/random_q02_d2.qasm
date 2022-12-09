OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(3.1528565) q[0];
rz(0.98002865) q[1];
cx q[0],q[1];
