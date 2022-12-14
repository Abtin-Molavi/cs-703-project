OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6516701) q[11];
rz(5.468826) q[5];
cx q[8],q[11];
cx q[5],q[8];
rz(2.6464324) q[8];
