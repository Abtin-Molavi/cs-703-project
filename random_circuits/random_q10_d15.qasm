OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[9],q[5];
cx q[2],q[6];
rz(2.4717868) q[0];
cx q[8],q[7];
rz(1.2780674) q[4];
rz(5.6158742) q[3];
rz(2.7075323) q[1];
cx q[1],q[6];
cx q[4],q[7];
cx q[9],q[8];
cx q[2],q[3];
rz(5.5972354) q[0];
rz(1.8174958) q[5];
rz(5.1095166) q[3];
cx q[1],q[9];
rz(2.5153477) q[5];
rz(3.4354933) q[7];
cx q[4],q[6];
rz(0.61180937) q[0];
cx q[2],q[8];
rz(1.1108417) q[0];
cx q[4],q[7];
rz(1.8272043) q[5];
rz(1.5541492) q[2];
cx q[3],q[1];
rz(2.238) q[8];
rz(5.5843135) q[6];
rz(3.9978031) q[9];
cx q[2],q[7];
rz(0.25777802) q[3];
rz(0.78754386) q[0];
rz(0.20870937) q[8];
cx q[9],q[5];
cx q[6],q[1];
rz(1.0748764) q[4];
cx q[3],q[2];
cx q[1],q[7];
rz(4.3555906) q[8];
rz(3.4946316) q[9];
rz(1.867778) q[5];
rz(2.8147479) q[0];
rz(5.4433459) q[4];
rz(0.46642652) q[6];
cx q[4],q[8];
rz(0.56429644) q[3];
rz(2.4853936) q[7];
cx q[1],q[6];
cx q[0],q[9];
rz(3.2756822) q[5];
rz(1.3395622) q[2];
rz(5.4957761) q[5];
cx q[8],q[0];
rz(5.4818455) q[6];
cx q[9],q[3];
cx q[4],q[2];
rz(3.7145776) q[7];
rz(4.2031555) q[1];
cx q[9],q[1];
cx q[2],q[8];
cx q[7],q[0];
cx q[4],q[6];
cx q[3],q[5];
cx q[1],q[7];
rz(3.611108) q[8];
cx q[2],q[6];
cx q[9],q[0];
rz(2.0013806) q[3];
rz(4.3066195) q[5];
rz(0.56630108) q[4];
cx q[1],q[3];
rz(3.6669571) q[5];
cx q[2],q[4];
cx q[6],q[8];
rz(0.97406441) q[9];
rz(0.71322414) q[7];
rz(5.8424443) q[0];
rz(4.978576) q[2];
cx q[0],q[6];
cx q[9],q[8];
cx q[1],q[7];
rz(0.79401502) q[5];
cx q[4],q[3];
rz(4.0172535) q[7];
cx q[1],q[8];
rz(2.1373749) q[0];
rz(3.2988996) q[4];
cx q[9],q[3];
rz(3.923374) q[2];
rz(1.2208094) q[5];
rz(2.9828769) q[6];
cx q[1],q[9];
cx q[3],q[6];
cx q[7],q[8];
rz(0.25625099) q[5];
rz(5.5883222) q[0];
cx q[2],q[4];
rz(2.3599647) q[1];
rz(0.13332149) q[3];
rz(4.6801243) q[8];
cx q[0],q[9];
cx q[4],q[2];
rz(5.6381109) q[6];
rz(5.6837166) q[7];
rz(1.2378985) q[5];
