OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.30036071) q[21];
rz(1.954893) q[23];
rz(0.58942049) q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[23],q[21];
rz(2.3847807) q[25];
rz(0.73540975) q[26];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[23],q[24];
cx q[21],q[23];
rz(3.4750648) q[24];
rz(5.2232997) q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
rz(0.16240961) q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[21];
rz(3.8805797) q[21];
rz(2.9339068) q[23];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[26];