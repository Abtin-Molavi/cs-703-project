OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(4.430614) q[20];
rz(0.22672869) q[20];
rz(2.9039641) q[20];
rz(4.9215463) q[20];
rz(0.64397432) q[20];
cx q[20],q[19];
rz(1.7013253) q[19];
rz(1.5022083) q[19];
rz(2.8160167) q[22];
rz(1.1181299) q[24];
rz(0.42686948) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[25],q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[22],q[25];
rz(0.81687126) q[25];
cx q[24],q[25];
rz(1.7791641) q[25];
rz(4.2530549) q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[22];
cx q[22],q[19];
cx q[25],q[24];
rz(1.8584339) q[25];
rz(3.0284875) q[25];
rz(5.8793245) q[25];