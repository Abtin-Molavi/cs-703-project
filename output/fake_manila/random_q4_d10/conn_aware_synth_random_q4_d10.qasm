OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
rz(0.75853612) q[1];
rz(6.0670025) q[1];
rz(1.2716335) q[1];
rz(6.0334233) q[1];
rz(5.0449546) q[1];
rz(0.13356237) q[1];
rz(3.0750501) q[1];
rz(3.2011787) q[1];
rz(5.0793938) q[3];
rz(4.7496402) q[4];
rz(0.69480734) q[4];
rz(1.4224081) q[3];
rz(0.048678178) q[3];
rz(5.5513311) q[3];
cx q[3],q[4];
rz(5.9529623) q[4];
rz(2.439189) q[4];
rz(0.20741427) q[4];
rz(1.9594288) q[4];
rz(3.3879717) q[4];
cx q[2],q[3];
cx q[3],q[4];
cx q[2],q[3];
cx q[1],q[2];
cx q[3],q[4];
cx q[2],q[3];
rz(4.6960203) q[3];
rz(2.7207697) q[3];
rz(5.2926491) q[3];
rz(4.4956037) q[2];
rz(2.5289664) q[3];
rz(5.7730006) q[3];
rz(4.0918269) q[2];