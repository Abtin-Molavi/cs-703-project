OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[9],q[8];
cx q[4],q[2];
cx q[14],q[11];
cx q[7],q[15];
rz(5.7890754) q[5];
rz(1.5805587) q[0];
rz(5.4829496) q[10];
rz(5.3631185) q[13];
rz(3.7784004) q[1];
rz(4.4640453) q[6];
rz(2.0875126) q[3];
rz(5.559703) q[12];
cx q[8],q[0];
rz(1.7754198) q[1];
rz(2.510533) q[13];
cx q[4],q[15];
cx q[5],q[6];
cx q[14],q[10];
cx q[2],q[9];
rz(3.1926663) q[3];
rz(1.4092984) q[7];
rz(4.5978006) q[11];
rz(0.54763511) q[12];
rz(1.0157589) q[15];
rz(5.9279808) q[6];
rz(6.0767473) q[3];
rz(3.4590212) q[13];
rz(5.6906094) q[11];
rz(0.041685327) q[2];
rz(0.37817402) q[7];
cx q[10],q[1];
cx q[14],q[5];
cx q[9],q[12];
cx q[8],q[4];
rz(6.0615443) q[0];
rz(4.205286) q[8];
cx q[11],q[14];
rz(5.7342064) q[7];
rz(1.2406396) q[4];
rz(5.1682587) q[10];
cx q[13],q[5];
rz(1.8217782) q[15];
rz(2.0573992) q[1];
rz(2.2554001) q[3];
cx q[2],q[0];
rz(5.6134662) q[12];
rz(2.7610572) q[9];
rz(6.1566303) q[6];
cx q[13],q[8];
rz(5.6190942) q[3];
cx q[0],q[6];
cx q[2],q[5];
rz(5.0094285) q[11];
rz(5.4403054) q[15];
rz(2.528475) q[1];
cx q[7],q[14];
rz(5.4959696) q[9];
rz(3.6013925) q[4];
cx q[12],q[10];
cx q[15],q[4];
rz(0.75842855) q[11];
cx q[7],q[0];
rz(0.64272931) q[5];
rz(1.3804446) q[8];
cx q[6],q[12];
rz(0.48751807) q[2];
rz(3.2526895) q[14];
cx q[3],q[1];
cx q[13],q[10];
rz(3.6073344) q[9];
rz(0.22669321) q[15];
cx q[5],q[1];
cx q[11],q[14];
rz(5.2780401) q[3];
cx q[6],q[2];
cx q[4],q[13];
rz(2.0683927) q[12];
rz(2.4763689) q[0];
cx q[10],q[7];
cx q[9],q[8];
rz(3.1801749) q[6];
rz(0.56858973) q[12];
cx q[5],q[8];
rz(2.1184939) q[2];
rz(1.459794) q[1];
rz(3.6986094) q[3];
cx q[9],q[13];
cx q[0],q[11];
rz(6.0774969) q[7];
rz(3.5001519) q[4];
rz(1.527152) q[15];
cx q[10],q[14];
rz(0.8581709) q[5];
rz(3.2977576) q[3];
rz(2.2810788) q[9];
rz(4.1110821) q[7];
rz(4.8955644) q[14];
cx q[13],q[2];
cx q[15],q[1];
cx q[10],q[8];
cx q[12],q[11];
cx q[4],q[0];
rz(1.1954918) q[6];
rz(2.7601326) q[6];
cx q[9],q[8];
rz(4.8326369) q[12];
cx q[3],q[13];
rz(6.2123533) q[15];
rz(4.057711) q[11];
rz(0.12106393) q[7];
cx q[0],q[1];
rz(5.6951066) q[2];
rz(5.525509) q[10];
cx q[4],q[14];
rz(4.0918518) q[5];