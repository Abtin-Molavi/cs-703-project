OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(0.15620787) q[1];
rz(3.7495669) q[0];
cx q[2],q[3];
