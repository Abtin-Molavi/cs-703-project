OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
rz(1.5228721) q[5];
rz(5.6189227) q[4];
rz(5.3150027) q[11];
cx q[10],q[0];
cx q[2],q[3];
rz(6.2327532) q[7];
cx q[1],q[12];
rz(4.1081683) q[6];
cx q[8],q[9];
rz(1.0898808) q[0];
rz(3.3806963) q[11];
rz(2.5143106) q[8];
rz(0.62571904) q[12];
cx q[9],q[1];
cx q[7],q[4];
cx q[3],q[6];
rz(1.6390017) q[10];
rz(5.0794348) q[5];
rz(0.37622429) q[2];
cx q[12],q[7];
rz(0.25941662) q[2];
rz(2.2834035) q[6];
cx q[11],q[3];
cx q[9],q[8];
cx q[10],q[0];
rz(3.6944166) q[5];
rz(2.0184962) q[4];
rz(1.3535281) q[1];
rz(1.2957782) q[5];
cx q[0],q[8];
cx q[10],q[6];
rz(2.94606) q[11];
cx q[1],q[2];
cx q[4],q[3];
cx q[7],q[9];
rz(4.9153862) q[12];
rz(4.1988288) q[2];
rz(3.7816665) q[12];
rz(2.4438321) q[1];
cx q[8],q[3];
cx q[9],q[11];
cx q[10],q[7];
rz(4.2023348) q[5];
cx q[6],q[0];
rz(2.1096976) q[4];
cx q[3],q[1];
rz(4.4739077) q[11];
rz(5.2041737) q[2];
rz(1.5181593) q[6];
rz(3.4241846) q[0];
cx q[4],q[8];
rz(3.0460222) q[7];
cx q[9],q[12];
cx q[10],q[5];
rz(0.23631725) q[5];
rz(3.3627248) q[6];
cx q[1],q[3];
cx q[10],q[12];
cx q[8],q[2];
rz(0.4007778) q[9];
cx q[7],q[4];
cx q[0],q[11];
