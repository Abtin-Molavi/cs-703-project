OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.37493069) q[5];
rz(5.257961) q[5];
rz(5.7645189) q[5];
rz(4.651824) q[8];
rz(1.2258368) q[11];
rz(3.89789) q[5];
cx q[11],q[8];
rz(2.6352801) q[8];
cx q[5],q[8];
cx q[11],q[8];
cx q[8],q[11];
