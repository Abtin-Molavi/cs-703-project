OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.8137776) q[0];
rz(1.5842995) q[2];
rz(4.1074649) q[3];
rz(6.2779009) q[4];
rz(2.9044888) q[5];
rz(1.9913955) q[7];
cx q[6],q[1];