OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.9805959) q[3];
cx q[3],q[2];
rz(2.2058108) q[2];
rz(2.111909) q[4];
rz(3.9920765) q[4];
rz(0.22525602) q[22];
cx q[25],q[24];
rz(4.0577908) q[24];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[22],q[25];
cx q[25],q[26];