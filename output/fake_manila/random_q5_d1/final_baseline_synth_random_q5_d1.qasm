OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[0],q[1];
cx q[3],q[2];
rz(0.088425022) q[4];
