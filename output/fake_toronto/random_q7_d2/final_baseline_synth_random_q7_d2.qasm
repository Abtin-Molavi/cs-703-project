OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.2051128) q[14];
cx q[14],q[13];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
rz(0.74916761) q[16];
cx q[14],q[16];
cx q[14],q[13];
rz(1.6592644) q[19];
rz(5.7432998) q[25];
cx q[25],q[22];
cx q[22],q[19];
