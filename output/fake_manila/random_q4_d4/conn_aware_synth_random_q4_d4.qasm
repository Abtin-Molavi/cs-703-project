OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(6.0651742) q[0];
rz(5.2995805) q[2];
rz(6.0913851) q[3];
rz(5.6438902) q[3];
cx q[1],q[0];
cx q[2],q[1];
cx q[3],q[2];
cx q[1],q[0];
cx q[2],q[1];
rz(1.8979209) q[1];
rz(5.8997412) q[1];
cx q[0],q[1];
rz(1.949207) q[0];
rz(1.4599476) q[2];