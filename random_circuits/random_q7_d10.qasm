OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
rz(2.2952555) q[5];
cx q[6],q[4];
cx q[0],q[1];
cx q[3],q[2];
cx q[0],q[2];
cx q[3],q[5];
rz(5.1912575) q[6];
rz(2.6877792) q[1];
rz(3.3389335) q[4];
rz(5.0950365) q[6];
rz(1.2805883) q[0];
rz(4.7752374) q[1];
cx q[5],q[2];
cx q[3],q[4];
rz(1.3045144) q[1];
cx q[3],q[5];
cx q[0],q[6];
rz(2.3772931) q[2];
rz(1.2036936) q[4];
rz(4.1979884) q[0];
rz(5.1176364) q[2];
cx q[5],q[4];
rz(3.5791673) q[3];
rz(0.53944019) q[1];
rz(3.776808) q[6];
rz(1.8040791) q[4];
cx q[0],q[3];
cx q[5],q[6];
rz(5.4351239) q[1];
rz(3.1632777) q[2];
rz(5.7983394) q[3];
cx q[0],q[4];
cx q[1],q[5];
rz(3.1811945) q[2];
rz(1.142984) q[6];
cx q[5],q[2];
cx q[4],q[3];
rz(1.003149) q[0];
cx q[6],q[1];
rz(1.0053622) q[6];
rz(0.47202508) q[0];
cx q[5],q[1];
rz(2.9984789) q[4];
cx q[3],q[2];
cx q[3],q[4];
cx q[1],q[6];
cx q[0],q[5];
rz(1.1428841) q[2];