OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.9068139) q[21];
rz(0.71451027) q[22];
rz(3.7979413) q[22];
rz(5.4114933) q[24];
rz(0.93130049) q[24];
rz(4.532036) q[25];
rz(5.0500868) q[25];
cx q[25],q[24];
rz(4.1739424) q[24];
rz(3.5851684) q[24];
cx q[25],q[26];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[21];
cx q[23],q[24];
cx q[21],q[23];
cx q[23],q[21];
cx q[21],q[23];
rz(3.3692424) q[26];
rz(1.2717878) q[26];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[22],q[25];
rz(0.045001161) q[24];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[25],q[22];
cx q[22],q[25];
cx q[24],q[25];
rz(1.1695008) q[24];
rz(5.884613) q[24];