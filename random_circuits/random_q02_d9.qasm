OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[0],q[1];
rz(3.656799) q[0];
rz(2.0503142) q[1];
rz(1.7021277) q[1];
rz(3.5125428) q[0];
rz(5.486547) q[1];
rz(0.35241326) q[0];
rz(3.7703734) q[0];
rz(2.4184384) q[1];
cx q[1],q[0];