OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[5],q[0];
rz(4.5049571) q[6];
rz(5.8033458) q[3];
cx q[4],q[7];
rz(2.9942933) q[9];
rz(3.2972707) q[2];
cx q[1],q[8];
cx q[8],q[5];
rz(3.0248666) q[3];
cx q[0],q[2];
rz(2.3392749) q[4];
rz(2.5811432) q[1];
cx q[7],q[9];
rz(2.9784987) q[6];
rz(0.72171735) q[5];
rz(6.2486223) q[0];
cx q[4],q[7];
cx q[6],q[1];
cx q[2],q[9];
rz(0.69825933) q[8];
rz(2.710302) q[3];
rz(1.9576481) q[7];
cx q[0],q[6];
cx q[9],q[5];
cx q[2],q[8];
cx q[3],q[1];
rz(5.7167331) q[4];
