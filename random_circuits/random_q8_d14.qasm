OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
cx q[7],q[2];
cx q[3],q[1];
cx q[0],q[4];
rz(1.5812442) q[5];
rz(5.4293928) q[6];
cx q[7],q[3];
cx q[4],q[2];
cx q[6],q[0];
rz(6.2277108) q[1];
rz(3.7845045) q[5];
cx q[0],q[6];
cx q[2],q[7];
rz(0.95525014) q[1];
rz(1.3950441) q[5];
rz(5.9673423) q[3];
rz(3.6083639) q[4];
cx q[1],q[3];
cx q[7],q[2];
rz(0.22166652) q[0];
cx q[5],q[6];
rz(5.0209589) q[4];
cx q[5],q[4];
cx q[6],q[2];
rz(1.7765264) q[0];
cx q[1],q[7];
rz(5.2082178) q[3];
cx q[3],q[7];
rz(0.44738499) q[1];
cx q[6],q[4];
cx q[2],q[0];
rz(5.4379233) q[5];
rz(0.066890467) q[2];
cx q[5],q[0];
cx q[1],q[4];
rz(4.9744065) q[7];
cx q[3],q[6];
rz(0.028256554) q[7];
cx q[0],q[5];
rz(2.3086422) q[2];
cx q[3],q[1];
rz(2.008886) q[4];
rz(5.8414423) q[6];
cx q[0],q[2];
cx q[7],q[6];
cx q[4],q[3];
cx q[1],q[5];
rz(5.6011838) q[2];
cx q[6],q[0];
cx q[4],q[5];
rz(5.9183932) q[7];
cx q[3],q[1];
rz(0.58692057) q[2];
rz(4.5717995) q[3];
cx q[4],q[0];
cx q[1],q[6];
cx q[5],q[7];
rz(1.4184456) q[2];
rz(1.8206115) q[0];
rz(0.65576851) q[6];
rz(6.0271072) q[4];
cx q[3],q[5];
rz(1.5874302) q[7];
rz(3.9847301) q[1];
rz(2.6554396) q[7];
cx q[2],q[0];
rz(5.1589128) q[5];
rz(2.3890201) q[4];
cx q[6],q[1];
rz(1.029693) q[3];
cx q[1],q[3];
cx q[2],q[7];
cx q[0],q[6];
rz(0.18496213) q[5];
rz(4.8240072) q[4];