OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.3446829) q[0];
rz(5.2361768) q[8];
rz(2.4658367) q[9];
rz(2.1152855) q[0];
rz(6.1334552) q[0];
rz(0.55264205) q[1];
rz(4.9398525) q[1];
rz(0.50370357) q[8];
rz(1.0568265) q[3];
rz(1.9892379) q[3];
rz(5.9826982) q[3];
cx q[8],q[11];
cx q[3],q[5];
cx q[3],q[2];
cx q[5],q[8];
rz(3.8302667) q[8];
cx q[2],q[1];
cx q[5],q[8];
cx q[9],q[8];
cx q[8],q[5];
rz(0.18573086) q[2];
rz(4.0880357) q[1];
rz(5.999277) q[1];
rz(5.4011736) q[2];
rz(6.0568022) q[2];
rz(0.37474905) q[8];
rz(2.1157657) q[11];
rz(4.6414507) q[11];
