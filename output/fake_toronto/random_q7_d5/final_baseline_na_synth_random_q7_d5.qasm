OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.6941851) q[0];
rz(0.73234407) q[0];
rz(2.8269524) q[0];
rz(5.6667261) q[19];
rz(0.70835949) q[21];
rz(0.37661713) q[21];
rz(2.4341547) q[22];
rz(0.41215772) q[22];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(5.0130684) q[26];
rz(0.79030579) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
rz(0.98842896) q[23];
rz(2.4793377) q[23];
rz(6.0247295) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[22];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[25];
cx q[19],q[22];
rz(3.7063118) q[19];
rz(1.3956488) q[25];