OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(3.773203) q[4];
rz(3.6904972) q[1];
rz(5.4838375) q[1];
cx q[3],q[2];
cx q[4],q[3];
rz(0.65867831) q[2];
