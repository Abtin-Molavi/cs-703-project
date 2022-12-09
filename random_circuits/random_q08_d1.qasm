OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
cx q[5],q[6];
rz(1.2010193) q[2];
cx q[0],q[3];
cx q[1],q[7];
rz(5.8758952) q[4];
