OPENQASM 2.0;
include "qelib1.inc";
qreg q[27];
rz(5.8672628) q[0];
rz(5.2646964) q[0];
rz(3.7951887) q[0];
rz(2.0060558) q[1];
rz(3.8060682) q[1];
rz(0.73468205) q[1];
rz(3.3313077) q[2];
cx q[3],q[5];
cx q[3],q[2];
cx q[2],q[3];
rz(3.3101563) q[5];
rz(3.1644893) q[5];
cx q[19],q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[19],q[20];
rz(0.26136329) q[20];
rz(0.080164595) q[20];
rz(2.9965474) q[25];
rz(1.9157338) q[25];
cx q[25],q[24];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[25],q[22];
cx q[22],q[19];
rz(4.3658624) q[22];
cx q[22],q[19];
cx q[19],q[22];
cx q[22],q[19];
cx q[22],q[25];
cx q[24],q[25];
rz(2.960214) q[24];