OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
rz(6.1305888) q[0];
rz(3.8007508) q[5];
rz(2.6637661) q[6];
cx q[8],q[3];
rz(1.7982927) q[2];
rz(3.7580355) q[7];
cx q[9],q[1];
rz(3.1338087) q[4];
rz(6.1933652) q[6];
cx q[5],q[2];
cx q[7],q[1];
cx q[0],q[3];
rz(3.6509133) q[8];
cx q[4],q[9];
rz(2.9085084) q[6];
cx q[8],q[5];
rz(4.0387823) q[3];
cx q[9],q[4];
rz(5.5080873) q[0];
rz(4.4386267) q[7];
rz(4.1554384) q[1];
rz(0.27676078) q[2];
rz(5.2008926) q[8];
rz(5.9039064) q[7];
rz(1.1938225) q[2];
rz(4.6143293) q[9];
cx q[6],q[5];
cx q[0],q[3];
cx q[4],q[1];
cx q[0],q[6];
rz(0.68434219) q[9];
cx q[1],q[3];
rz(3.2950576) q[2];
rz(1.6511549) q[5];
rz(2.8128949) q[8];
rz(5.4638397) q[4];
rz(3.7315747) q[7];
cx q[0],q[1];
cx q[3],q[5];
cx q[4],q[8];
cx q[6],q[2];
rz(1.0199113) q[7];
rz(4.6400971) q[9];
cx q[5],q[8];
rz(3.9288534) q[6];
cx q[3],q[0];
cx q[4],q[9];
rz(4.9868005) q[1];
rz(0.29720195) q[2];
rz(0.16474737) q[7];
cx q[0],q[4];
cx q[8],q[3];
rz(4.5418727) q[9];
rz(2.2358652) q[7];
cx q[5],q[6];
rz(1.6062268) q[2];
rz(2.9285414) q[1];
rz(1.3184314) q[5];
rz(1.8041975) q[9];
cx q[3],q[6];
cx q[8],q[1];
rz(0.10962429) q[2];
cx q[4],q[7];
rz(0.64142845) q[0];
cx q[2],q[6];
cx q[5],q[1];
cx q[0],q[3];
rz(0.99574407) q[9];
cx q[4],q[7];
rz(3.8762275) q[8];
rz(1.7684029) q[4];
cx q[9],q[7];
cx q[3],q[1];
cx q[2],q[6];
cx q[5],q[0];
rz(0.28261273) q[8];
