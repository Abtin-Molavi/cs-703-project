OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.4879656) q[4];
rz(5.5767173) q[5];
rz(2.7088232) q[6];
rz(1.0077838) q[7];
rz(1.5452397) q[8];
cx q[1],q[3];
cx q[0],q[2];