OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.0382222) q[3];
rz(5.0358269) q[3];
rz(5.8400137) q[7];
rz(5.6420987) q[11];
rz(3.8510407) q[12];
cx q[12],q[10];
cx q[7],q[10];
cx q[13],q[14];
cx q[14],q[11];
cx q[8],q[11];
cx q[11],q[8];
cx q[8],q[11];
cx q[14],q[16];
cx q[16],q[14];
cx q[14],q[16];
cx q[14],q[11];
rz(3.2092343) q[11];
cx q[14],q[13];
rz(5.3551915) q[24];
rz(4.5787183) q[25];
cx q[25],q[24];
