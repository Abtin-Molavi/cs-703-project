OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.3455298) q[3];
rz(4.6173458) q[6];
cx q[2],q[5];
cx q[7],q[1];
cx q[8],q[4];
cx q[0],q[9];
