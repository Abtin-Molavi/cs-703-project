OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.6067552) q[3];
rz(4.9973646) q[4];
rz(2.2640547) q[5];
rz(3.9870915) q[9];
rz(3.6454002) q[10];
rz(4.4822114) q[11];
rz(0.20867819) q[12];
rz(0.93523516) q[15];
cx q[7],q[14];
cx q[2],q[0];
cx q[8],q[1];
cx q[13],q[6];
