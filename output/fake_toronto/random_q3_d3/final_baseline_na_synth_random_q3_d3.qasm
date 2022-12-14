OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6516701) q[24];
cx q[25],q[24];
rz(5.468826) q[26];
cx q[26],q[25];
rz(2.6464324) q[25];
