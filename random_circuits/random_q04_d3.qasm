OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
rz(0.43074062) q[2];
rz(3.0428663) q[3];
rz(5.34026) q[0];
rz(5.5330765) q[1];
rz(0.57601936) q[2];
rz(0.46955348) q[3];
rz(1.0636909) q[0];
rz(4.4294282) q[1];
cx q[2],q[1];
cx q[0],q[3];