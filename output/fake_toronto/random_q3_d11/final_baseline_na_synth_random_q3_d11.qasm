OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.6935904) q[24];
rz(5.3286877) q[24];
rz(5.1150935) q[25];
rz(1.1012128) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
rz(5.6251229) q[26];
rz(6.075245) q[26];
cx q[26],q[25];
rz(4.3262686) q[25];
cx q[25],q[24];
rz(1.1125972) q[24];
rz(3.641275) q[24];
rz(5.4441093) q[24];
rz(2.8322983) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[26],q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[24],q[25];
rz(0.42237029) q[25];
rz(2.3738416) q[25];
