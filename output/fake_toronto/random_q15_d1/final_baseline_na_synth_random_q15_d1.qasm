OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.5395865) q[0];
rz(5.9405374) q[1];
cx q[3],q[2];
rz(1.4929886) q[4];
rz(3.0379315) q[5];
rz(5.3755089) q[6];
cx q[8],q[11];
cx q[20],q[19];
cx q[21],q[18];
cx q[24],q[25];
