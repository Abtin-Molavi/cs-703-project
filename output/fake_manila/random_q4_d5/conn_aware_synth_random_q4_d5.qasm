OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(2.6977628) q[2];
rz(4.8382051) q[3];
rz(4.5540724) q[0];
rz(4.1027253) q[0];
rz(5.5229917) q[0];
rz(2.356346) q[1];
rz(0.56522392) q[2];
rz(5.8304046) q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[2],q[3];
cx q[3],q[2];
