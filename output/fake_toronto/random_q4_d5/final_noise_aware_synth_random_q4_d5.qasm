OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.6977628) q[8];
rz(4.8382051) q[11];
rz(4.5540724) q[15];
rz(4.1027253) q[15];
rz(5.5229917) q[15];
rz(2.356346) q[5];
rz(0.56522392) q[8];
rz(5.8304046) q[8];
cx q[5],q[8];
cx q[8],q[5];
cx q[8],q[11];
cx q[11],q[8];
