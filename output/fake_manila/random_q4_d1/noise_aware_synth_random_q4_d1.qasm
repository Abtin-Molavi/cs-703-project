OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(3.7495669) q[1];
rz(0.15620787) q[2];
cx q[3],q[4];
