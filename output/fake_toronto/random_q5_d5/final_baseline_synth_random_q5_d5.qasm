OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.1441532) q[19];
rz(2.0058929) q[22];
cx q[19],q[22];
cx q[19],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(2.8694193) q[25];
cx q[22],q[25];
rz(4.3597247) q[25];
cx q[22],q[25];
cx q[25],q[22];
cx q[22],q[25];
rz(5.3670225) q[26];
rz(6.1014656) q[26];
cx q[25],q[26];
rz(0.33595685) q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[22];
cx q[22],q[19];
cx q[19],q[16];
rz(5.3376567) q[16];
rz(0.76871907) q[19];