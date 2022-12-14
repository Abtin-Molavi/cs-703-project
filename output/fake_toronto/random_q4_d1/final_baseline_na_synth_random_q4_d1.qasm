OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.7495669) q[0];
rz(0.15620787) q[1];
cx q[24],q[25];
