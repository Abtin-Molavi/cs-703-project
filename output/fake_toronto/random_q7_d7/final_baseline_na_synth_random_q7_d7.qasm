OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.4003225) q[16];
rz(4.1437014) q[16];
rz(2.1045027) q[16];
rz(3.1304322) q[16];
rz(1.9278874) q[16];
rz(2.1621331) q[19];
rz(1.1671405) q[19];
rz(0.83224948) q[19];
rz(1.4343148) q[19];
cx q[16],q[19];
rz(1.5730782) q[20];
cx q[19],q[20];
rz(3.1958857) q[20];
rz(5.9885149) q[20];
rz(3.7196041) q[20];
rz(4.0132137) q[20];
rz(4.5377175) q[22];
rz(2.6379651) q[24];
cx q[25],q[22];
cx q[22],q[19];
rz(3.3877289) q[22];
rz(4.9113246) q[22];
rz(1.5035952) q[26];
rz(5.9625579) q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[26],q[25];
rz(4.537337) q[26];
rz(6.1613769) q[26];
rz(5.2303788) q[26];
