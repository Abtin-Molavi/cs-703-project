OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(0.17105978) q[2];
rz(3.6627338) q[1];
cx q[1],q[0];
rz(3.8893253) q[0];
rz(0.33171116) q[0];
cx q[1],q[2];
rz(0.0019100289) q[2];
rz(1.8063778) q[2];
cx q[1],q[0];