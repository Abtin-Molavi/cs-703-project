OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[3],q[5];
rz(3.1226084) q[4];
rz(2.1435941) q[1];
rz(5.5169846) q[0];
rz(4.8830383) q[2];
