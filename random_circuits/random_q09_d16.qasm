OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5],q[7];
rz(0.43189202) q[4];
rz(4.1402151) q[3];
cx q[1],q[0];
rz(0.31499493) q[6];
rz(5.9036368) q[2];
rz(2.2936415) q[8];
cx q[0],q[6];
cx q[8],q[1];
cx q[4],q[3];
rz(4.873363) q[2];
cx q[7],q[5];
cx q[0],q[8];
cx q[7],q[6];
rz(5.394646) q[3];
rz(5.7372167) q[4];
cx q[2],q[5];
rz(5.2975392) q[1];
rz(4.2589681) q[6];
rz(3.4061602) q[2];
cx q[5],q[1];
cx q[0],q[7];
rz(2.4029475) q[4];
cx q[8],q[3];
rz(2.1394103) q[4];
cx q[0],q[2];
rz(2.084561) q[5];
cx q[3],q[7];
rz(0.54009464) q[1];
rz(2.366542) q[8];
rz(3.9057017) q[6];
cx q[4],q[0];
cx q[2],q[3];
rz(2.5503961) q[1];
rz(1.7607613) q[5];
cx q[8],q[6];
rz(5.5890294) q[7];
rz(1.9244137) q[2];
cx q[4],q[6];
cx q[8],q[0];
rz(5.1781062) q[1];
rz(2.6431237) q[5];
cx q[3],q[7];
rz(5.1030061) q[7];
cx q[2],q[3];
rz(4.2156994) q[1];
rz(2.6099023) q[6];
cx q[8],q[4];
cx q[0],q[5];
rz(1.6352329) q[7];
rz(6.127439) q[6];
cx q[5],q[3];
rz(3.0050591) q[0];
rz(4.6589128) q[2];
rz(4.1583536) q[8];
cx q[1],q[4];
rz(4.4037571) q[8];
cx q[0],q[6];
rz(3.1050095) q[7];
cx q[2],q[4];
rz(1.0054771) q[1];
cx q[3],q[5];
cx q[8],q[3];
rz(0.32927823) q[6];
rz(5.389663) q[2];
cx q[4],q[0];
rz(1.4912969) q[7];
cx q[5],q[1];
rz(1.2002158) q[7];
rz(2.5940846) q[8];
rz(5.9769653) q[6];
rz(2.7982969) q[1];
cx q[0],q[3];
rz(3.4090264) q[4];
cx q[2],q[5];
cx q[2],q[4];
rz(2.744754) q[1];
cx q[5],q[3];
rz(4.9887573) q[0];
rz(5.1653042) q[7];
cx q[8],q[6];
cx q[0],q[7];
rz(5.7894702) q[4];
rz(1.6948155) q[3];
rz(4.2625839) q[8];
cx q[5],q[6];
cx q[1],q[2];
cx q[0],q[2];
rz(3.4570024) q[5];
rz(2.0580942) q[8];
cx q[1],q[6];
rz(0.75534924) q[3];
rz(5.0388347) q[7];
rz(3.7447178) q[4];
rz(5.6187724) q[3];
cx q[7],q[4];
rz(5.2310829) q[1];
rz(0.37802525) q[6];
cx q[0],q[5];
cx q[2],q[8];