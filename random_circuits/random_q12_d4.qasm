OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
cx q[9],q[1];
cx q[10],q[5];
cx q[11],q[4];
rz(2.2682454) q[3];
cx q[7],q[8];
rz(4.6438464) q[2];
rz(2.2883427) q[6];
rz(5.2745173) q[0];
rz(6.1592957) q[9];
cx q[1],q[4];
rz(1.5614974) q[8];
cx q[10],q[7];
rz(0.43782986) q[2];
rz(3.7522764) q[3];
rz(4.6398998) q[0];
rz(2.3048648) q[6];
rz(2.0108042) q[11];
rz(2.1236317) q[5];
rz(3.0266907) q[10];
rz(5.997106) q[0];
rz(0.72590656) q[7];
rz(1.0765246) q[9];
rz(4.2207189) q[11];
cx q[3],q[1];
cx q[6],q[2];
cx q[4],q[8];
rz(0.69587498) q[5];
rz(0.92378241) q[8];
cx q[6],q[1];
cx q[4],q[2];
rz(3.8492137) q[11];
rz(4.4529305) q[7];
cx q[9],q[0];
rz(5.4448) q[3];
cx q[5],q[10];
