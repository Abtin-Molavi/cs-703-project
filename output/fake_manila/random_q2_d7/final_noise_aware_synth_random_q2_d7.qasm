OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(4.9419302) q[3];
rz(0.42392532) q[4];
rz(5.1623415) q[3];
rz(4.6265828) q[3];
cx q[3],q[4];
rz(2.2671061) q[4];
rz(2.7820234) q[4];
cx q[4],q[3];
