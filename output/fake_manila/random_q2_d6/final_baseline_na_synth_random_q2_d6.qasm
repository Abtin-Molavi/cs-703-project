OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(5.0545483) q[4];
rz(4.1705958) q[4];
cx q[3],q[4];
rz(3.8927641) q[4];
rz(2.2373516) q[4];
cx q[4],q[3];
