OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.6592644) q[5];
rz(5.7432998) q[9];
rz(0.74916761) q[3];
rz(1.2051128) q[0];
cx q[2],q[1];
cx q[9],q[8];
cx q[2],q[3];
cx q[0],q[1];
cx q[8],q[5];
