OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(1.7221845) q[4];
rz(5.2271186) q[4];
cx q[4],q[3];
rz(5.9106233) q[3];
rz(4.0002107) q[3];
