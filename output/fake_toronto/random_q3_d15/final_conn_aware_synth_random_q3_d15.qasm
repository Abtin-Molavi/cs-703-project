OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(3.1669645) q[4];
rz(1.5051091) q[4];
rz(4.2693431) q[7];
rz(3.7031025) q[7];
rz(3.3621755) q[7];
rz(1.6380762) q[1];
rz(2.7700469) q[1];
rz(5.2243719) q[1];
rz(1.3259757) q[1];
rz(6.0825385) q[1];
rz(4.5766503) q[4];
rz(3.8352997) q[1];
rz(1.6234292) q[4];
cx q[1],q[4];
rz(0.50816428) q[4];
rz(2.9773797) q[4];
rz(0.26928765) q[4];
rz(4.8623235) q[4];
rz(4.3966128) q[4];
rz(2.4529805) q[4];
cx q[7],q[4];
rz(1.9456395) q[4];
rz(2.4264578) q[4];
cx q[4],q[1];
rz(4.5244784) q[4];
rz(5.9573491) q[4];
cx q[1],q[4];
cx q[4],q[7];
rz(3.8969669) q[7];
rz(4.9480124) q[1];