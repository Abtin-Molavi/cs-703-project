OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
rz(1.7289991) q[5];
cx q[1],q[2];
cx q[4],q[0];
rz(0.067695512) q[6];
cx q[3],q[7];
cx q[5],q[6];
rz(1.1487929) q[2];
rz(1.9070664) q[0];
cx q[3],q[1];
cx q[4],q[7];
cx q[3],q[7];
rz(0.75379812) q[6];
rz(0.92764574) q[1];
rz(5.01727) q[0];
cx q[2],q[5];
rz(4.3828253) q[4];
rz(2.1955682) q[4];
rz(6.2168062) q[3];
rz(2.2141278) q[0];
cx q[6],q[1];
cx q[2],q[7];
rz(2.6022505) q[5];
rz(4.5086497) q[4];
cx q[3],q[7];
cx q[2],q[6];
cx q[1],q[0];
rz(1.3760992) q[5];
cx q[0],q[4];
cx q[2],q[3];
rz(3.1659786) q[6];
rz(2.6036653) q[7];
cx q[1],q[5];
rz(0.18926528) q[5];
rz(3.9488979) q[4];
rz(4.5396649) q[1];
rz(1.2046981) q[2];
rz(5.801211) q[7];
cx q[3],q[6];
rz(4.6050633) q[0];
rz(1.7475143) q[7];
rz(1.2306384) q[3];
rz(0.20768646) q[6];
cx q[0],q[1];
cx q[2],q[4];
rz(6.2224417) q[5];
cx q[4],q[7];
rz(5.797145) q[6];
cx q[3],q[0];
rz(0.12594071) q[1];
rz(0.58612278) q[5];
rz(3.4351795) q[2];
cx q[5],q[4];
rz(0.85155369) q[6];
cx q[7],q[0];
rz(1.522256) q[3];
rz(2.7176371) q[1];
rz(0.26787314) q[2];
rz(1.9785504) q[1];
rz(0.86553444) q[2];
rz(2.6225194) q[4];
rz(5.4768456) q[0];
cx q[5],q[6];
cx q[3],q[7];
