OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[5],q[0];
cx q[1],q[4];
cx q[2],q[3];
rz(4.4470158) q[2];
rz(3.109626) q[3];
rz(5.5784181) q[4];
cx q[5],q[0];
rz(0.49088793) q[1];
cx q[3],q[1];
rz(1.3666519) q[5];
cx q[4],q[0];
rz(4.4657131) q[2];
