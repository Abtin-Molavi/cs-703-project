OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.3255271) q[26];
rz(4.4316874) q[24];
rz(4.5174343) q[25];
rz(5.6608036) q[24];
rz(1.5749893) q[22];
rz(0.66235728) q[26];
rz(0.39798678) q[25];
cx q[26],q[25];
cx q[25],q[22];
rz(0.55673703) q[22];
cx q[26],q[25];
cx q[25],q[26];
rz(1.0013972) q[22];
cx q[25],q[22];
cx q[24],q[25];
cx q[25],q[24];
cx q[25],q[26];
rz(2.4221548) q[25];
rz(1.9281156) q[26];
rz(2.7936838) q[26];
rz(2.5244535) q[26];
rz(2.6513068) q[26];
rz(2.1934949) q[22];
rz(2.6349726) q[25];
rz(0.8065518) q[25];
rz(2.6695754) q[25];
