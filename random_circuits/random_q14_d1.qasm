OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
rz(5.3515759) q[5];
rz(0.87807065) q[1];
rz(5.7702744) q[8];
cx q[10],q[13];
rz(2.063846) q[4];
cx q[11],q[12];
cx q[6],q[2];
rz(6.139186) q[0];
rz(1.7908102) q[9];
cx q[7],q[3];