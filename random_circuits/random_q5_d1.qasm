OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[4],q[1];
rz(0.088425022) q[0];
cx q[2],q[3];
