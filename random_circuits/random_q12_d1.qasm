OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
rz(5.3320828) q[5];
rz(1.3263772) q[8];
rz(3.1277169) q[2];
cx q[10],q[11];
rz(1.3533428) q[9];
rz(5.7888243) q[3];
cx q[1],q[4];
cx q[0],q[7];
rz(5.7599542) q[6];
