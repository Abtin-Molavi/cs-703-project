OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.1277169) q[0];
rz(5.7888243) q[23];
rz(5.3320828) q[19];
rz(5.7599542) q[21];
rz(1.3263772) q[22];
rz(1.3533428) q[16];
cx q[8],q[11];
cx q[7],q[6];
cx q[25],q[24];
