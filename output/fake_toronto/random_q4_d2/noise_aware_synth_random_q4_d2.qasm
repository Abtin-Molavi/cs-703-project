OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.773203) q[5];
rz(3.6904972) q[2];
rz(5.4838375) q[2];
cx q[8],q[11];
cx q[5],q[8];
rz(0.65867831) q[11];