OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(1.9871325) q[0];
rz(2.3071966) q[0];
rz(5.0827044) q[1];
rz(3.5931429) q[1];
rz(2.9180844) q[3];
rz(4.5671804) q[4];
rz(6.2683214) q[5];
rz(3.5819712) q[5];
rz(1.7542524) q[5];
rz(3.8486947) q[5];
rz(3.2376121) q[5];
cx q[4],q[0];
cx q[5],q[4];
cx q[3],q[4];
cx q[0],q[3];
cx q[4],q[3];
cx q[2],q[0];
cx q[3],q[1];
rz(2.4741723) q[1];
cx q[4],q[2];
cx q[0],q[4];
cx q[2],q[1];
rz(5.0999808) q[4];
rz(5.4551614) q[4];
rz(6.0687359) q[4];
rz(2.3871707) q[2];
