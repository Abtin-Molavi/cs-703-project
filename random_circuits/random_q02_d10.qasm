OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rz(3.3667277) q[0];
rz(3.2409923) q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
rz(2.2373545) q[0];
rz(1.1553032) q[1];
rz(5.1490576) q[1];
rz(1.3680362) q[0];
rz(3.1197019) q[0];
rz(4.327451) q[1];
cx q[0],q[1];
rz(0.73242561) q[1];
rz(1.7844439) q[0];
cx q[1],q[0];