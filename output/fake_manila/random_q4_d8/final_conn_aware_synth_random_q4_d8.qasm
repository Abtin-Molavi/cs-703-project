OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(3.3255271) q[1];
rz(4.4316874) q[3];
rz(4.5174343) q[4];
rz(5.6608036) q[3];
rz(1.5749893) q[2];
rz(0.66235728) q[1];
rz(0.39798678) q[4];
cx q[1],q[2];
cx q[3],q[4];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[3],q[2];
cx q[4],q[3];
rz(2.4221548) q[4];
rz(2.6349726) q[4];
rz(0.8065518) q[4];
rz(2.6695754) q[4];
rz(1.0013972) q[3];
rz(0.55673703) q[3];
cx q[2],q[3];
rz(1.9281156) q[1];
rz(2.7936838) q[1];
rz(2.5244535) q[1];
rz(2.6513068) q[1];
rz(2.1934949) q[2];