OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(5.6516701) q[2];
rz(5.468826) q[0];
cx q[1],q[2];
cx q[0],q[1];
rz(2.6464324) q[1];
