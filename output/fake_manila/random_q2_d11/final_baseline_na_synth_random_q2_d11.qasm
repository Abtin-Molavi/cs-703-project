OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(0.31407796) q[3];
rz(1.7427065) q[3];
rz(2.7456224) q[3];
rz(2.3670376) q[3];
cx q[3],q[4];
rz(5.346812) q[4];
rz(3.9645561) q[4];
rz(2.6908642) q[4];
rz(1.2275453) q[4];
cx q[4],q[3];
cx q[3],q[4];
