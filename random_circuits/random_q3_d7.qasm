OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[1],q[2];
rz(3.0904863) q[0];
cx q[2],q[0];
rz(3.6617792) q[1];
rz(5.0894228) q[2];
rz(6.0373865) q[0];
rz(6.2019138) q[1];
cx q[2],q[0];
rz(0.082429618) q[1];
rz(5.5218641) q[2];
cx q[1],q[0];
rz(2.4992064) q[2];
rz(6.0767666) q[1];
rz(0.41072197) q[0];
cx q[0],q[1];
rz(5.5168599) q[2];
