OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5],q[7];
cx q[3],q[8];
rz(4.396339) q[2];
rz(4.610348) q[0];
rz(3.1978252) q[4];
rz(2.1411406) q[1];
rz(6.1277344) q[6];
cx q[1],q[4];
rz(1.2511796) q[5];
cx q[2],q[6];
rz(4.073399) q[7];
cx q[0],q[3];
rz(0.47820417) q[8];
rz(1.2105426) q[0];
rz(5.4498674) q[1];
rz(4.1764378) q[7];
rz(2.0099722) q[2];
rz(3.4675769) q[6];
cx q[5],q[4];
rz(5.7518877) q[8];
rz(5.2219022) q[3];
rz(4.0503997) q[8];
rz(4.3236179) q[2];
rz(5.3795571) q[6];
rz(2.3954493) q[5];
rz(1.7619506) q[7];
rz(0.085162971) q[3];
cx q[1],q[4];
rz(4.4747374) q[0];
rz(0.052021247) q[5];
cx q[6],q[1];
rz(2.566695) q[7];
cx q[0],q[8];
rz(0.23973232) q[4];
cx q[3],q[2];
rz(5.8930025) q[4];
rz(5.4204495) q[5];
cx q[3],q[7];
cx q[8],q[1];
rz(4.9893011) q[0];
cx q[2],q[6];
cx q[1],q[7];
rz(0.77660889) q[6];
rz(4.5022999) q[4];
rz(1.9670471) q[5];
rz(0.29986239) q[8];
cx q[0],q[2];
rz(2.8796207) q[3];
cx q[4],q[3];
rz(5.1819084) q[0];
cx q[7],q[2];
rz(2.5762789) q[8];
rz(4.5881852) q[6];
cx q[5],q[1];
rz(3.6693411) q[5];
rz(3.2115734) q[7];
cx q[6],q[4];
cx q[2],q[1];
cx q[0],q[8];
rz(1.6414122) q[3];
cx q[1],q[7];
cx q[6],q[8];
rz(5.743292) q[0];
rz(5.7281751) q[4];
rz(1.0571116) q[3];
rz(4.3554519) q[5];
rz(4.6152886) q[2];
rz(4.6703159) q[7];
cx q[6],q[8];
rz(1.5259541) q[2];
rz(0.22647192) q[0];
cx q[1],q[5];
rz(6.2092902) q[3];
rz(0.27809878) q[4];
cx q[6],q[5];
cx q[3],q[4];
rz(6.2022699) q[0];
cx q[8],q[2];
rz(0.35783012) q[7];
rz(2.6056667) q[1];