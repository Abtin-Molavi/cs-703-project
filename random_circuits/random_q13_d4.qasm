OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
rz(4.2629898) q[0];
rz(6.2692625) q[7];
cx q[5],q[11];
cx q[2],q[8];
rz(5.0400022) q[4];
rz(1.3534057) q[1];
rz(6.2358939) q[9];
rz(1.1819261) q[3];
cx q[6],q[12];
rz(1.227006) q[10];
rz(5.2401139) q[1];
cx q[8],q[11];
cx q[0],q[5];
cx q[12],q[3];
cx q[7],q[9];
rz(4.1703334) q[6];
cx q[10],q[2];
rz(1.0316889) q[4];
cx q[5],q[4];
rz(2.4672947) q[0];
rz(0.56218362) q[6];
cx q[7],q[11];
rz(1.669649) q[1];
rz(1.4779163) q[2];
rz(2.860995) q[10];
rz(3.5633382) q[9];
rz(0.89674554) q[8];
cx q[3],q[12];
cx q[10],q[6];
cx q[11],q[12];
cx q[0],q[2];
rz(3.3912879) q[1];
cx q[4],q[5];
cx q[3],q[8];
rz(4.1145971) q[9];
rz(4.5203836) q[7];