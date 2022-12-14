OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.8491831) q[3];
rz(1.0553079) q[5];
cx q[0],q[5];
cx q[0],q[4];
cx q[3],q[4];
