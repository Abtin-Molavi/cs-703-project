OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rz(0.37493069) q[1];
cx q[0],q[2];
rz(2.6352801) q[2];
rz(5.257961) q[1];
rz(4.651824) q[0];
cx q[2],q[0];
rz(5.7645189) q[1];
cx q[1],q[2];
rz(1.2258368) q[0];
cx q[2],q[0];
rz(3.89789) q[1];