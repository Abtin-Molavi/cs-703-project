OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(6.0651742) q[17];
rz(5.2995805) q[21];
rz(6.0913851) q[23];
rz(5.6438902) q[23];
cx q[18],q[17];
cx q[21],q[18];
cx q[18],q[17];
cx q[23],q[21];
cx q[21],q[18];
rz(1.8979209) q[18];
rz(5.8997412) q[18];
cx q[17],q[18];
rz(1.949207) q[17];
rz(1.4599476) q[21];
