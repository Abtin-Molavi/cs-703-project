OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
cx q[1],q[5];
cx q[4],q[3];
rz(6.0614381) q[2];
rz(1.4288624) q[0];
rz(2.7852788) q[4];
rz(4.7508041) q[3];
rz(5.1801027) q[1];
cx q[2],q[5];
rz(4.8006436) q[0];
rz(4.6541999) q[4];
cx q[1],q[3];
cx q[0],q[5];
rz(4.1531011) q[2];
cx q[1],q[5];
cx q[3],q[0];
rz(4.9773043) q[4];
rz(5.6310178) q[2];
cx q[1],q[2];
cx q[4],q[0];
cx q[3],q[5];
rz(4.1397652) q[2];
rz(0.15858601) q[3];
cx q[5],q[4];
rz(4.5045959) q[1];
rz(4.3483706) q[0];