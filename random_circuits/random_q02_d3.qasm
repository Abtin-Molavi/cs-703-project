OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(5.7429907) q[0];
rz(0.19151085) q[1];
rz(1.191463) q[0];
rz(0.41415416) q[1];
cx q[0],q[1];
