OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.5395865) q[12];
rz(5.9405374) q[17];
rz(1.4929886) q[4];
rz(3.0379315) q[1];
rz(5.3755089) q[0];
cx q[19],q[20];
cx q[3],q[2];
cx q[15],q[18];
cx q[6],q[7];
cx q[21],q[23];
