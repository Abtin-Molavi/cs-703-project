OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(3.6617792) q[3];
rz(6.2019138) q[3];
rz(0.082429618) q[3];
rz(6.0767666) q[3];
rz(3.0904863) q[4];
cx q[3],q[4];
cx q[4],q[3];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[4],q[3];
rz(6.0373865) q[3];
cx q[2],q[3];
rz(5.0894228) q[3];
rz(5.5218641) q[3];
rz(2.4992064) q[3];
rz(5.5168599) q[3];
rz(0.41072197) q[4];