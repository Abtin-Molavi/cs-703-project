OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.6060217) q[5];
cx q[10],q[12];
cx q[11],q[14];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[13],q[14];
