OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
rz(4.9025044) q[1];
rz(2.986546) q[9];
cx q[4],q[5];
cx q[7],q[10];
rz(4.6760112) q[2];
cx q[8],q[6];
rz(4.5289925) q[3];
rz(1.3761712) q[0];
cx q[6],q[8];
rz(2.6456068) q[7];
cx q[4],q[0];
rz(0.68444041) q[5];
rz(3.409037) q[2];
cx q[1],q[3];
rz(0.45919706) q[9];
rz(2.7964003) q[10];
rz(1.7293623) q[6];
rz(4.3009058) q[5];
cx q[4],q[8];
cx q[3],q[7];
cx q[0],q[9];
cx q[2],q[1];
rz(6.0149394) q[10];
cx q[7],q[9];
cx q[1],q[3];
rz(2.0468282) q[5];
cx q[4],q[8];
rz(3.0838176) q[6];
rz(2.9768138) q[2];
rz(0.80238367) q[0];
rz(2.4380644) q[10];
rz(1.27424) q[4];
cx q[7],q[0];
cx q[8],q[5];
rz(1.1307843) q[6];
cx q[3],q[10];
cx q[1],q[2];
rz(1.8351791) q[9];
rz(4.8554559) q[1];
cx q[6],q[3];
rz(1.4939706) q[4];
cx q[10],q[2];
rz(0.28592484) q[7];
rz(4.2048201) q[8];
rz(5.5038391) q[9];
rz(4.7026573) q[0];
rz(1.1826132) q[5];