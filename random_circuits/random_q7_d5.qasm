OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[2],q[1];
cx q[0],q[6];
rz(2.4341547) q[5];
rz(5.0130684) q[3];
rz(0.70835949) q[4];
rz(0.41215772) q[5];
cx q[1],q[2];
cx q[4],q[3];
cx q[0],q[6];
cx q[2],q[5];
rz(0.37661713) q[4];
rz(0.6941851) q[0];
rz(5.6667261) q[6];
cx q[3],q[1];
rz(0.98842896) q[1];
rz(3.7063118) q[5];
cx q[6],q[2];
cx q[3],q[4];
rz(0.73234407) q[0];
rz(1.3956488) q[2];
cx q[5],q[6];
rz(0.79030579) q[4];
rz(2.4793377) q[1];
rz(6.0247295) q[3];
rz(2.8269524) q[0];
