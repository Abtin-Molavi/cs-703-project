OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.0553079) q[10];
cx q[12],q[10];
cx q[12],q[15];
rz(2.8491831) q[18];
cx q[18],q[15];
