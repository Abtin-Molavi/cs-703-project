OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.0807361) q[24];
rz(4.3964485) q[25];
rz(5.5382454) q[26];
rz(3.9430169) q[26];
rz(0.96202999) q[26];
rz(0.52256812) q[26];
rz(1.9535677) q[26];
cx q[25],q[26];
rz(4.5537473) q[26];
rz(1.9791802) q[26];
rz(5.8282229) q[26];
rz(0.84906078) q[26];
cx q[25],q[26];
cx q[25],q[24];
rz(6.2804539) q[24];
rz(0.50737806) q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[26],q[25];
rz(2.6507364) q[25];
rz(4.3339962) q[25];
rz(4.3549174) q[25];
