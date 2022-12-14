OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.7221845) q[24];
rz(5.2271186) q[24];
cx q[24],q[25];
rz(5.9106233) q[25];
rz(4.0002107) q[25];
