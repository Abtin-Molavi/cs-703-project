OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(5.1197973) q[3];
cx q[1],q[2];
rz(5.1322718) q[4];
rz(6.2611587) q[0];
rz(2.2451891) q[4];
rz(1.5607893) q[3];
rz(3.7448123) q[1];
cx q[0],q[2];
rz(5.359995) q[3];
rz(1.352167) q[0];
cx q[1],q[4];
rz(2.8096096) q[2];
rz(5.734897) q[1];
cx q[2],q[4];
rz(5.4038658) q[0];
rz(0.21293217) q[3];
rz(4.5253871) q[3];
cx q[1],q[4];
rz(4.5836805) q[2];
rz(0.21893013) q[0];
cx q[2],q[4];
rz(0.42429332) q[0];
cx q[1],q[3];
rz(0.60075922) q[1];
rz(4.2770673) q[2];
cx q[3],q[4];
rz(4.7338201) q[0];
rz(5.843238) q[1];
cx q[0],q[3];
cx q[4],q[2];
rz(5.8731032) q[0];
rz(2.8138424) q[4];
rz(3.5452333) q[2];
cx q[3],q[1];
