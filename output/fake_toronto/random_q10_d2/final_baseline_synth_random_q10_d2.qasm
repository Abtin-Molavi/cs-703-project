OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
cx q[1],q[4];
cx q[1],q[0];
rz(4.6188958) q[0];
rz(3.3974445) q[5];
rz(2.790067) q[5];
cx q[4],q[7];
cx q[7],q[4];
cx q[4],q[7];
cx q[10],q[7];
rz(0.92145154) q[12];
cx q[10],q[12];
rz(0.51006319) q[18];
cx q[18],q[21];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[18],q[17];
rz(1.3317275) q[17];
cx q[21],q[18];