OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rz(4.0227386) q[5];
rz(0.072846787) q[4];
rz(1.1235891) q[0];
cx q[3],q[1];
cx q[7],q[2];
rz(2.3818612) q[6];
rz(4.1752711) q[1];
rz(3.9134877) q[4];
rz(6.0863656) q[7];
cx q[2],q[5];
cx q[0],q[3];
rz(5.1564397) q[6];
cx q[3],q[1];
cx q[4],q[7];
rz(0.79900221) q[0];
rz(2.9836358) q[5];
cx q[6],q[2];
cx q[7],q[0];
rz(5.7599287) q[6];
rz(4.4435827) q[2];
cx q[4],q[3];
rz(1.7257977) q[5];
rz(4.6882847) q[1];
rz(2.7470589) q[3];
cx q[7],q[0];
cx q[1],q[5];
cx q[4],q[2];
rz(0.11419607) q[6];
cx q[0],q[3];
rz(5.121824) q[1];
rz(4.825632) q[2];
cx q[5],q[6];
rz(5.6482035) q[7];
rz(1.7912828) q[4];
rz(1.8275384) q[1];
cx q[4],q[0];
cx q[3],q[2];
rz(4.9022967) q[7];
cx q[6],q[5];
cx q[0],q[5];
cx q[1],q[3];
cx q[4],q[7];
rz(4.0402549) q[2];
rz(0.13414745) q[6];
cx q[4],q[5];
rz(2.8215954) q[6];
cx q[7],q[0];
cx q[1],q[3];
rz(5.5961916) q[2];
rz(3.4800576) q[7];
cx q[2],q[0];
rz(0.88325548) q[5];
cx q[4],q[1];
rz(3.5184547) q[6];
rz(5.3007051) q[3];
cx q[5],q[3];
cx q[1],q[0];
cx q[6],q[7];
cx q[4],q[2];
rz(3.811797) q[2];
cx q[0],q[7];
cx q[6],q[4];
rz(4.1989514) q[5];
rz(0.74523712) q[1];
rz(2.237382) q[3];