OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
rz(3.8482453) q[4];
rz(3.4649371) q[1];
cx q[3],q[5];
rz(4.0818632) q[0];
rz(5.3468934) q[2];
rz(2.332042) q[3];
cx q[4],q[1];
cx q[2],q[0];
rz(4.0632707) q[5];
cx q[0],q[2];
rz(2.0026109) q[1];
cx q[3],q[4];
rz(3.625499) q[5];
rz(1.4169219) q[3];
cx q[4],q[5];
cx q[2],q[0];
rz(2.580122) q[1];
rz(5.544158) q[5];
cx q[4],q[1];
cx q[0],q[3];
rz(5.2616675) q[2];
cx q[4],q[0];
cx q[5],q[1];
rz(0.81557982) q[2];
rz(1.8112077) q[3];
rz(4.9452341) q[4];
rz(3.532651) q[1];
rz(1.193942) q[0];
cx q[5],q[3];
rz(0.41929407) q[2];
rz(2.2601805) q[5];
rz(3.3450702) q[2];
cx q[1],q[3];
cx q[0],q[4];
rz(5.5576224) q[4];
rz(6.2258167) q[2];
cx q[3],q[1];
cx q[0],q[5];
cx q[0],q[1];
rz(0.58305558) q[4];
cx q[3],q[5];
rz(1.6813014) q[2];
cx q[0],q[5];
rz(0.56062666) q[2];
rz(2.6527562) q[3];
rz(3.4598412) q[4];
rz(5.6720344) q[1];
cx q[0],q[2];
rz(2.526748) q[5];
rz(1.5771654) q[3];
cx q[1],q[4];
rz(0.39427415) q[0];
cx q[4],q[3];
cx q[2],q[1];
rz(4.8454679) q[5];
cx q[5],q[1];
cx q[2],q[4];
rz(3.7969655) q[0];
rz(1.4142104) q[3];
rz(5.3069665) q[5];
cx q[0],q[1];
rz(0.81419899) q[4];
rz(2.9773161) q[2];
rz(2.3340869) q[3];
