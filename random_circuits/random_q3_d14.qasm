OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[2],q[1];
rz(3.0139127) q[0];
rz(4.4260656) q[2];
cx q[1],q[0];
rz(4.5924874) q[1];
cx q[2],q[0];
rz(5.9897648) q[2];
cx q[0],q[1];
cx q[0],q[1];
rz(2.6460316) q[2];
rz(5.0470584) q[2];
cx q[1],q[0];
rz(2.213998) q[1];
cx q[2],q[0];
rz(0.032373014) q[0];
cx q[1],q[2];
cx q[0],q[2];
rz(3.6562783) q[1];
cx q[2],q[0];
rz(2.7868691) q[1];
rz(3.1112997) q[2];
rz(4.7349315) q[1];
rz(2.5114349) q[0];
cx q[1],q[0];
rz(2.9008882) q[2];
cx q[1],q[2];
rz(1.359472) q[0];
rz(0.58770693) q[1];
rz(4.9333214) q[2];
rz(0.35037647) q[0];
