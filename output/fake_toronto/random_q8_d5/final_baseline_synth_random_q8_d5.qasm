OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.3446829) q[5];
rz(2.1152855) q[5];
rz(6.1334552) q[5];
rz(5.2361768) q[11];
rz(0.50370357) q[11];
rz(1.0568265) q[13];
rz(1.9892379) q[13];
rz(5.9826982) q[13];
cx q[13],q[12];
cx q[11],q[14];
cx q[11],q[8];
cx q[13],q[14];
rz(3.8302667) q[14];
rz(2.1157657) q[8];
rz(4.6414507) q[8];
rz(2.4658367) q[16];
cx q[16],q[14];
cx q[11],q[14];
cx q[14],q[11];
cx q[11],q[14];
cx q[16],q[14];
rz(0.37474905) q[14];
rz(0.55264205) q[18];
rz(4.9398525) q[18];
cx q[15],q[18];
cx q[18],q[15];
cx q[15],q[18];
cx q[12],q[15];
rz(0.18573086) q[12];
rz(5.4011736) q[12];
rz(6.0568022) q[12];
rz(4.0880357) q[15];
rz(5.999277) q[15];
