OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
cx q[5],q[7];
rz(3.3446829) q[0];
cx q[6],q[4];
rz(5.2361768) q[1];
rz(0.55264205) q[3];
rz(2.4658367) q[2];
cx q[2],q[0];
cx q[1],q[5];
rz(0.18573086) q[4];
cx q[6],q[7];
rz(4.9398525) q[3];
cx q[2],q[0];
rz(0.50370357) q[1];
cx q[4],q[3];
cx q[5],q[7];
rz(1.0568265) q[6];
rz(2.1152855) q[0];
rz(5.4011736) q[4];
rz(3.8302667) q[7];
cx q[2],q[1];
rz(2.1157657) q[5];
rz(4.0880357) q[3];
rz(1.9892379) q[6];
rz(0.37474905) q[1];
rz(4.6414507) q[5];
rz(6.0568022) q[4];
cx q[2],q[7];
rz(5.9826982) q[6];
rz(5.999277) q[3];
rz(6.1334552) q[0];
