OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
cx q[3],q[8];
cx q[5],q[7];
rz(1.6698124) q[1];
rz(5.1949394) q[10];
rz(4.7965401) q[2];
rz(2.625203) q[0];
cx q[4],q[9];
rz(0.14449028) q[6];
cx q[8],q[10];
cx q[3],q[0];
cx q[5],q[1];
rz(3.1967457) q[2];
rz(3.2883623) q[6];
cx q[9],q[7];
rz(5.5982322) q[4];
rz(0.51991555) q[1];
cx q[10],q[8];
cx q[5],q[0];
rz(0.64315902) q[3];
cx q[6],q[2];
cx q[4],q[9];
rz(3.7729958) q[7];
rz(1.696189) q[9];
rz(0.43578771) q[8];
cx q[7],q[3];
rz(3.9639261) q[0];
rz(5.971038) q[5];
cx q[10],q[2];
rz(3.4420542) q[1];
rz(6.1592647) q[4];
rz(1.7880803) q[6];
cx q[8],q[4];
cx q[9],q[3];
cx q[10],q[1];
cx q[6],q[2];
rz(3.7170366) q[5];
rz(4.0323543) q[0];
rz(5.7719242) q[7];
rz(1.7572051) q[1];
cx q[10],q[0];
cx q[3],q[8];
cx q[9],q[4];
rz(1.8627892) q[7];
cx q[6],q[2];
rz(0.036805317) q[5];
cx q[2],q[5];
cx q[0],q[10];
cx q[9],q[7];
cx q[3],q[1];
rz(0.57874811) q[4];
cx q[8],q[6];
cx q[10],q[5];
cx q[7],q[3];
rz(4.0176157) q[4];
rz(0.27327467) q[1];
cx q[0],q[8];
rz(2.3920611) q[9];
cx q[6],q[2];
