OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
cx q[4],q[5];
rz(2.2378601) q[2];
cx q[3],q[0];
rz(3.4679525) q[1];
rz(3.6878165) q[6];
rz(4.5830785) q[0];
rz(2.3021198) q[5];
cx q[6],q[2];
cx q[3],q[1];
rz(6.167583) q[4];
cx q[6],q[0];
cx q[1],q[5];
rz(0.21782512) q[3];
rz(5.6106851) q[2];
rz(0.96684437) q[4];
rz(5.8360348) q[5];
rz(4.2768163) q[6];
rz(3.2082106) q[4];
rz(3.5694423) q[1];
cx q[0],q[3];
rz(3.4524438) q[2];
cx q[5],q[6];
rz(2.7529976) q[0];
cx q[3],q[2];
cx q[1],q[4];
rz(0.91557931) q[5];
rz(3.5498432) q[0];
cx q[1],q[2];
cx q[6],q[4];
rz(2.2059497) q[3];
rz(2.0556871) q[2];
rz(4.0444602) q[4];
cx q[1],q[0];
cx q[5],q[3];
rz(5.3499066) q[6];
cx q[3],q[5];
rz(5.2018473) q[4];
cx q[0],q[1];
rz(3.859306) q[6];
rz(4.2880791) q[2];
cx q[5],q[6];
rz(2.8803885) q[4];
rz(5.4991395) q[3];
cx q[1],q[2];
rz(5.3928803) q[0];
cx q[6],q[5];
rz(6.141404) q[2];
rz(1.3252962) q[0];
rz(4.0455937) q[4];
cx q[3],q[1];
cx q[6],q[0];
rz(2.8195928) q[2];
cx q[1],q[5];
cx q[4],q[3];
cx q[6],q[3];
cx q[4],q[5];
cx q[2],q[0];
rz(0.66492481) q[1];
cx q[5],q[2];
cx q[1],q[4];
rz(6.0383745) q[6];
cx q[3],q[0];
cx q[3],q[2];
rz(0.47581578) q[5];
rz(4.7869752) q[6];
rz(0.95720539) q[4];
rz(0.11539297) q[0];
rz(2.4815033) q[1];
