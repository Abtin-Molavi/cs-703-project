OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.3361647) q[1];
rz(5.1604294) q[0];
rz(0.40354425) q[4];
rz(1.3599796) q[11];
cx q[10],q[3];
cx q[1],q[5];
cx q[9],q[4];
cx q[6],q[10];
cx q[0],q[2];
cx q[11],q[5];
cx q[9],q[8];
cx q[6],q[7];
rz(4.418919) q[2];
rz(4.7298427) q[7];
rz(1.0468676) q[8];
rz(5.3662913) q[3];