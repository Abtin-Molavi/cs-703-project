OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.8491831) q[1];
rz(1.0553079) q[6];
cx q[1],q[4];
cx q[7],q[4];
cx q[7],q[6];
