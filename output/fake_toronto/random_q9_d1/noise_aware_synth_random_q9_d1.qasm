OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.4879656) q[0];
rz(5.5767173) q[15];
rz(2.7088232) q[4];
rz(1.0077838) q[5];
rz(1.5452397) q[20];
cx q[25],q[24];
cx q[11],q[8];