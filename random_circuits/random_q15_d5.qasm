OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
rz(0.70728716) q[6];
rz(3.1744341) q[11];
rz(3.8522157) q[7];
cx q[3],q[4];
rz(0.9931919) q[14];
cx q[2],q[0];
rz(2.048209) q[12];
rz(2.043324) q[1];
rz(5.357423) q[5];
rz(4.5988936) q[9];
cx q[10],q[8];
rz(3.9964728) q[13];
rz(3.6314393) q[10];
rz(2.8059081) q[8];
cx q[5],q[14];
cx q[9],q[12];
cx q[13],q[7];
rz(3.7685038) q[1];
cx q[0],q[2];
rz(2.9302723) q[4];
rz(0.70499193) q[3];
cx q[11],q[6];
rz(3.6813103) q[3];
cx q[9],q[1];
cx q[13],q[4];
rz(6.0351833) q[0];
cx q[5],q[10];
cx q[7],q[11];
rz(0.66284328) q[12];
rz(2.9120749) q[8];
cx q[2],q[6];
rz(2.342908) q[14];
cx q[7],q[12];
rz(2.1774095) q[0];
cx q[5],q[3];
cx q[6],q[10];
cx q[9],q[14];
rz(5.6667579) q[4];
rz(0.006969919) q[11];
rz(2.87863) q[1];
cx q[8],q[2];
rz(1.8660966) q[13];
cx q[9],q[2];
rz(2.1269737) q[11];
cx q[10],q[14];
cx q[12],q[13];
cx q[0],q[7];
cx q[6],q[4];
cx q[8],q[3];
cx q[5],q[1];
