OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.1150935) q[5];
rz(1.1012128) q[5];
rz(5.6251229) q[8];
rz(6.075245) q[8];
rz(5.6935904) q[11];
rz(5.3286877) q[11];
cx q[11],q[8];
rz(2.8322983) q[8];
rz(4.3262686) q[8];
cx q[5],q[8];
rz(3.641275) q[8];
cx q[8],q[11];
rz(1.1125972) q[8];
rz(5.4441093) q[8];
cx q[11],q[8];
cx q[8],q[5];
rz(0.42237029) q[11];
rz(2.3738416) q[11];
