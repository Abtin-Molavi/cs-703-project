OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.0545483) q[24];
rz(4.1705958) q[24];
cx q[25],q[24];
rz(3.8927641) q[24];
rz(2.2373516) q[24];
cx q[24],q[25];
