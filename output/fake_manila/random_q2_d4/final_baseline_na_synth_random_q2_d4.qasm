OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(2.8698915) q[3];
rz(1.7854145) q[4];
cx q[3],q[4];
rz(5.6479812) q[4];
rz(0.92336829) q[4];
cx q[4],q[3];