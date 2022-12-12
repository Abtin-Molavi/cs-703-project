OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(5.468826) q[1];
rz(5.6516701) q[3];
cx q[2],q[3];
cx q[1],q[2];
rz(2.6464324) q[2];
