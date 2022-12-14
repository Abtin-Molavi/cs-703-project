OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.7221845) q[0];
rz(5.2271186) q[0];
cx q[0],q[1];
rz(5.9106233) q[1];
rz(4.0002107) q[1];
