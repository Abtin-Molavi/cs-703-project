OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
cx q[5],q[9];
cx q[3],q[10];
cx q[4],q[2];
rz(4.4152438) q[7];
cx q[8],q[6];
cx q[1],q[0];
rz(4.0636351) q[8];
cx q[9],q[2];
cx q[1],q[7];
rz(1.8719268) q[5];
cx q[3],q[4];
rz(2.620842) q[6];
cx q[10],q[0];
cx q[5],q[10];
rz(3.5060768) q[4];
rz(5.1518046) q[7];
cx q[9],q[2];
rz(6.1813102) q[6];
cx q[3],q[0];
rz(3.1685993) q[1];
rz(0.73994215) q[8];
cx q[7],q[10];
rz(2.6486332) q[4];
cx q[6],q[5];
cx q[9],q[3];
cx q[0],q[2];
rz(5.2799765) q[8];
rz(2.4131095) q[1];
rz(1.1155625) q[0];
cx q[7],q[9];
rz(0.41513725) q[6];
rz(3.4749057) q[3];
rz(3.4458025) q[10];
rz(2.5550507) q[4];
cx q[2],q[1];
rz(1.6421273) q[5];
rz(5.6818867) q[8];
cx q[3],q[0];
cx q[10],q[2];
cx q[4],q[1];
rz(3.4778027) q[5];
rz(5.9981639) q[7];
rz(4.9617946) q[6];
cx q[9],q[8];
rz(0.76332473) q[2];
cx q[3],q[6];
rz(5.6469588) q[0];
rz(0.62464771) q[9];
rz(1.2284335) q[10];
rz(5.1868835) q[1];
rz(2.3827667) q[4];
cx q[8],q[5];
rz(4.6966952) q[7];
rz(1.7329414) q[1];
rz(3.4006114) q[6];
cx q[10],q[9];
rz(2.8181684) q[2];
rz(1.7880067) q[4];
cx q[8],q[0];
cx q[5],q[7];
rz(6.1197107) q[3];
cx q[8],q[10];
rz(0.29814251) q[6];
cx q[4],q[7];
rz(6.2150352) q[1];
rz(4.6016599) q[2];
cx q[5],q[0];
cx q[9],q[3];
cx q[0],q[9];
rz(1.4120089) q[6];
cx q[5],q[2];
cx q[1],q[3];
rz(5.6935202) q[10];
cx q[4],q[8];
rz(1.0321858) q[7];