OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[2],q[3];
rz(5.6361008) q[7];
cx q[4],q[7];
cx q[12],q[10];
rz(5.8738251) q[10];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[12],q[10];
cx q[10],q[12];
cx q[12],q[10];
cx q[10],q[12];
cx q[6],q[7];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[4];
cx q[7],q[10];
cx q[10],q[7];
cx q[7],q[10];
cx q[4],q[7];
cx q[1],q[4];
cx q[4],q[1];
cx q[1],q[4];
cx q[1],q[2];
cx q[3],q[2];
rz(2.1232398) q[3];
