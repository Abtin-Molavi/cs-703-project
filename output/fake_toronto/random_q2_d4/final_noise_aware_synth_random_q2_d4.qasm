OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.8698915) q[11];
rz(1.7854145) q[8];
cx q[11],q[8];
rz(5.6479812) q[8];
rz(0.92336829) q[8];
cx q[8],q[11];
