OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(5.4501073) q[8];
rz(5.7555108) q[5];
rz(6.2493463) q[0];
rz(0.54885012) q[9];
cx q[10],q[12];
cx q[7],q[1];
cx q[4],q[14];
rz(1.1302553) q[2];
cx q[13],q[11];
rz(0.60178081) q[15];
cx q[6],q[3];
rz(2.4027581) q[4];
cx q[15],q[13];
rz(2.296003) q[7];
rz(4.0844968) q[9];
cx q[8],q[14];
rz(5.3755385) q[6];
rz(2.844754) q[10];
rz(4.1152361) q[11];
rz(4.370421) q[2];
cx q[12],q[3];
rz(4.9428376) q[5];
cx q[1],q[0];
cx q[7],q[1];
cx q[15],q[11];
rz(4.6635847) q[3];
rz(4.0962921) q[0];
cx q[5],q[13];
rz(0.014806761) q[9];
cx q[2],q[6];
rz(5.5611057) q[10];
rz(2.3180842) q[14];
rz(3.3971536) q[12];
cx q[4],q[8];
cx q[14],q[8];
cx q[0],q[11];
rz(4.9332929) q[10];
cx q[2],q[9];
cx q[3],q[6];
rz(3.565561) q[7];
cx q[1],q[12];
cx q[5],q[4];
rz(2.2456257) q[15];
rz(6.1702227) q[13];
rz(2.5308192) q[3];
rz(4.2233699) q[4];
rz(3.449526) q[1];
cx q[6],q[7];
rz(4.7760867) q[10];
rz(4.442479) q[14];
rz(4.4790699) q[11];
cx q[13],q[12];
cx q[15],q[2];
rz(4.75272) q[0];
cx q[8],q[9];
rz(4.5509774) q[5];
rz(4.8368587) q[8];
cx q[4],q[11];
rz(6.2079703) q[12];
cx q[13],q[9];
cx q[1],q[3];
cx q[5],q[10];
cx q[15],q[7];
cx q[2],q[14];
cx q[0],q[6];
