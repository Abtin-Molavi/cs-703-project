OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(2.0058929) q[0];
rz(2.8694193) q[1];
rz(4.1441532) q[3];
rz(5.3670225) q[2];
rz(6.1014656) q[2];
cx q[3],q[0];
cx q[3],q[4];
cx q[0],q[1];
rz(4.3597247) q[1];
cx q[0],q[2];
cx q[2],q[1];
cx q[1],q[4];
cx q[4],q[3];
rz(0.33595685) q[0];
rz(5.3376567) q[3];
rz(0.76871907) q[4];
