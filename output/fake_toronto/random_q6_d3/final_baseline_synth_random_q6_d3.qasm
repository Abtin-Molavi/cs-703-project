OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.3666519) q[3];
rz(4.4470158) q[9];
rz(4.4657131) q[9];
cx q[9],q[8];
rz(0.49088793) q[11];
cx q[11],q[14];
cx q[14],q[13];
rz(5.5784181) q[14];
cx q[8],q[11];
rz(3.109626) q[8];
