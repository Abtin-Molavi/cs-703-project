OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(0.77407141) q[11];
rz(1.6325605) q[11];
rz(4.851002) q[11];
rz(2.5529458) q[11];
rz(0.20024484) q[12];
rz(4.7460216) q[12];
rz(2.2580734) q[13];
cx q[12],q[13];
rz(5.5588193) q[13];
rz(0.81681822) q[16];
rz(0.60211844) q[16];
rz(5.4543083) q[16];
rz(2.6174573) q[16];
rz(5.6193575) q[16];
rz(5.4399231) q[16];
rz(0.76098902) q[16];
cx q[16],q[14];
rz(4.8034742) q[14];
cx q[14],q[13];
cx q[11],q[14];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[14],q[11];
cx q[11],q[14];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[11];
cx q[13],q[14];
cx q[13],q[12];
rz(4.3190805) q[13];
rz(4.3786336) q[13];
