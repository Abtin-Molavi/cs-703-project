OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.6974752) q[15];
rz(3.3453614) q[16];
rz(4.3835842) q[9];
rz(6.0561172) q[1];
rz(3.0131068) q[4];
rz(0.8997262) q[4];
rz(2.3269647) q[3];
rz(4.673274) q[0];
rz(4.2624729) q[0];
cx q[1],q[2];
cx q[3],q[2];
cx q[9],q[8];
cx q[12],q[13];
cx q[11],q[8];
cx q[16],q[14];
cx q[15],q[12];
cx q[14],q[11];
rz(1.1628374) q[13];