OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cx q[1],q[0];
rz(0.37300412) q[0];
rz(1.6328052) q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
