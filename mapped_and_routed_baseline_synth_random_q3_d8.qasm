OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(5.5382454) q[0];
rz(3.9430169) q[0];
rz(0.96202999) q[0];
rz(0.52256812) q[0];
rz(1.9535677) q[0];
rz(4.3964485) q[1];
cx q[1],q[0];
rz(4.5537473) q[0];
rz(1.9791802) q[0];
rz(5.8282229) q[0];
rz(0.84906078) q[0];
cx q[1],q[0];
rz(4.0807361) q[2];
cx q[1],q[2];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
rz(6.2804539) q[2];
rz(0.50737806) q[2];
cx q[1],q[2];
rz(2.6507364) q[2];
rz(4.3339962) q[2];
rz(4.3549174) q[2];
