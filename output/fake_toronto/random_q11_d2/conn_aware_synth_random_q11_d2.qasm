OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.0382222) q[6];
rz(5.0358269) q[6];
rz(5.6420987) q[12];
rz(5.3551915) q[2];
rz(4.5787183) q[3];
rz(5.8400137) q[14];
rz(3.8510407) q[19];
cx q[21],q[23];
cx q[18],q[15];
cx q[21],q[18];
cx q[15],q[12];
cx q[19],q[16];
cx q[3],q[2];
cx q[14],q[16];
rz(3.2092343) q[23];
