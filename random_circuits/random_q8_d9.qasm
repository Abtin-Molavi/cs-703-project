OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rz(4.2121511) q[4];
cx q[5],q[3];
rz(3.637333) q[0];
rz(4.6649999) q[6];
rz(5.2614178) q[1];
rz(3.565888) q[7];
rz(2.6598869) q[2];
rz(3.4538759) q[5];
cx q[1],q[3];
cx q[2],q[4];
cx q[0],q[7];
rz(1.035026) q[6];
cx q[6],q[7];
cx q[0],q[1];
rz(5.9740657) q[5];
rz(4.2176639) q[4];
cx q[3],q[2];
cx q[2],q[0];
rz(1.9988662) q[6];
rz(1.1773789) q[3];
cx q[1],q[7];
cx q[4],q[5];
rz(3.0556438) q[3];
rz(5.0152854) q[7];
cx q[5],q[2];
rz(0.17000679) q[6];
cx q[0],q[4];
rz(5.9721274) q[1];
cx q[5],q[4];
rz(5.4931924) q[6];
cx q[7],q[0];
rz(5.4699039) q[2];
cx q[3],q[1];
cx q[3],q[6];
rz(1.8600536) q[4];
cx q[2],q[7];
rz(1.3718056) q[1];
rz(6.0914484) q[0];
rz(0.024071621) q[5];
rz(5.186361) q[7];
cx q[2],q[0];
cx q[4],q[3];
cx q[1],q[6];
rz(2.234102) q[5];
rz(4.1084316) q[0];
cx q[5],q[1];
rz(6.2505774) q[3];
cx q[4],q[2];
rz(2.1673446) q[6];
rz(6.0019584) q[7];
