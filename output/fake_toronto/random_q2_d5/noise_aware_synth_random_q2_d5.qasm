OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.7221845) q[11];
rz(5.2271186) q[11];
cx q[11],q[8];
rz(5.9106233) q[8];
rz(4.0002107) q[8];
