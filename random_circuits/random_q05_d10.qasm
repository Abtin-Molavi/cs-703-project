OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
cx q[2],q[1];
cx q[3],q[4];
rz(3.3164706) q[0];
cx q[0],q[2];
cx q[1],q[4];
rz(1.2553483) q[3];
cx q[3],q[1];
rz(4.3822093) q[2];
cx q[0],q[4];
rz(0.32196773) q[2];
rz(3.8290342) q[4];
cx q[1],q[3];
rz(2.2237898) q[0];
cx q[0],q[3];
cx q[2],q[1];
rz(6.0732163) q[4];
rz(1.3434967) q[4];
cx q[1],q[2];
rz(2.3761703) q[0];
rz(4.8569076) q[3];
rz(4.4450967) q[2];
rz(1.0238071) q[1];
cx q[4],q[3];
rz(1.3433821) q[0];
cx q[2],q[1];
rz(6.1790537) q[4];
cx q[0],q[3];
rz(1.2631784) q[1];
cx q[0],q[4];
rz(5.2334912) q[2];
rz(3.5316184) q[3];
cx q[1],q[2];
rz(6.2546616) q[3];
cx q[0],q[4];