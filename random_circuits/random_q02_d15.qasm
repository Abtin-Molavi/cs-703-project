OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cx q[1],q[0];
rz(2.0022847) q[0];
rz(3.2052373) q[1];
rz(5.4633549) q[0];
rz(5.6603094) q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[1],q[0];
cx q[1],q[0];
cx q[1],q[0];
rz(5.3180536) q[1];
rz(0.032257929) q[0];
rz(5.4156513) q[1];
rz(0.86767701) q[0];
cx q[0],q[1];
rz(4.5096399) q[0];
rz(3.5669508) q[1];
rz(1.4906268) q[1];
rz(6.2380591) q[0];
rz(0.54676766) q[1];
rz(5.6265156) q[0];
rz(2.2713031) q[0];
rz(3.6957368) q[1];