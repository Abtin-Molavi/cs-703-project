OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.8137776) q[17];
rz(1.5842995) q[20];
rz(4.1074649) q[6];
rz(6.2779009) q[7];
rz(2.9044888) q[5];
rz(1.9913955) q[9];
cx q[11],q[8];