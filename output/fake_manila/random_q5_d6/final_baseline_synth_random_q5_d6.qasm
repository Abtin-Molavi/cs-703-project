OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(0.30036071) q[0];
cx q[1],q[0];
rz(1.954893) q[2];
rz(0.58942049) q[2];
rz(2.3847807) q[3];
rz(0.73540975) q[4];
cx q[3],q[4];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
rz(3.4750648) q[2];
rz(5.2232997) q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[1];
rz(0.16240961) q[2];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(3.8805797) q[0];
rz(2.9339068) q[1];
cx q[3],q[4];
