OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[0],q[4];
rz(2.8491831) q[3];
cx q[1],q[2];
rz(1.0553079) q[5];
cx q[3],q[4];
cx q[0],q[5];
cx q[1],q[2];