OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
cx q[3],q[0];
rz(3.1914667) q[10];
cx q[1],q[6];
cx q[4],q[2];
rz(0.19068592) q[5];
rz(5.7443469) q[8];
rz(5.8099393) q[9];
cx q[11],q[7];
rz(2.0368945) q[6];
cx q[1],q[9];
cx q[2],q[11];
cx q[8],q[4];
cx q[7],q[0];
cx q[5],q[3];
rz(4.3119817) q[10];
cx q[3],q[0];
cx q[5],q[6];
rz(6.1621398) q[1];
cx q[11],q[8];
cx q[2],q[10];
cx q[9],q[7];
rz(1.8829199) q[4];
cx q[10],q[2];
rz(1.3394777) q[0];
rz(0.68791299) q[9];
cx q[1],q[11];
rz(2.4342233) q[7];
rz(2.1136121) q[5];
cx q[3],q[4];
cx q[8],q[6];
rz(4.14266) q[2];
rz(4.4911907) q[3];
rz(3.0065006) q[10];
rz(0.54359814) q[6];
rz(5.0238364) q[0];
cx q[7],q[8];
rz(2.1072303) q[11];
rz(1.3426696) q[5];
rz(1.4408638) q[9];
cx q[1],q[4];
cx q[6],q[8];
rz(2.2185805) q[10];
cx q[9],q[11];
rz(0.12729602) q[1];
rz(4.5401933) q[5];
cx q[0],q[4];
cx q[3],q[7];
rz(3.7082055) q[2];
rz(2.5968525) q[8];
rz(2.9961764) q[7];
cx q[10],q[1];
cx q[4],q[3];
rz(5.0291731) q[6];
cx q[2],q[5];
cx q[11],q[0];
rz(0.6515995) q[9];
cx q[2],q[6];
cx q[1],q[7];
cx q[0],q[9];
rz(2.1354146) q[3];
rz(2.6080702) q[4];
cx q[11],q[5];
cx q[10],q[8];
cx q[10],q[0];
cx q[8],q[9];
rz(4.9956715) q[4];
cx q[2],q[5];
rz(0.8010863) q[3];
rz(0.21603743) q[7];
rz(4.3658948) q[6];
rz(2.6315654) q[1];
rz(5.1794464) q[11];
cx q[3],q[4];
rz(3.0220285) q[2];
cx q[7],q[8];
rz(2.341995) q[6];
rz(0.0622181) q[10];
rz(3.9974187) q[1];
rz(6.231003) q[5];
cx q[9],q[0];
rz(1.4287107) q[11];
cx q[0],q[1];
cx q[3],q[9];
rz(0.76910897) q[7];
rz(1.7207892) q[5];
rz(1.206289) q[6];
rz(4.4068893) q[4];
cx q[10],q[11];
cx q[2],q[8];
cx q[6],q[11];
rz(1.5034463) q[0];
cx q[3],q[8];
cx q[1],q[10];
cx q[7],q[9];
rz(1.4362326) q[2];
rz(0.31189091) q[4];
rz(5.8628978) q[5];
rz(4.4607443) q[10];
cx q[3],q[0];
rz(2.1141596) q[8];
cx q[9],q[5];
rz(1.641884) q[4];
rz(1.44045) q[7];
rz(3.8062308) q[1];
cx q[11],q[2];
rz(5.5931727) q[6];
