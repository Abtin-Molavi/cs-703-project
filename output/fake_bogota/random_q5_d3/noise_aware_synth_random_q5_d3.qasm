OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(1.7831146) q[3];
rz(5.7433584) q[0];
cx q[2],q[1];
cx q[1],q[0];
cx q[3],q[4];
cx q[3],q[2];
cx q[2],q[1];
rz(6.2305009) q[4];
rz(0.0020672301) q[4];
rz(4.3090761) q[0];
